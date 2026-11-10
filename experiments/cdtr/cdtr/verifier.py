"""Clinical decision-equivalence verifier.

Question: within the (evidence-restricted) conformal set  C = {x : x consistent with evidence, F(x) <= T},  T = F*_0 + q,
does every hypothesis yield the same rule-derived response as the current best hypothesis x_hat?

For every alternative response class c != h(x_hat) we solve one mixed-integer feasibility problem (HiGHS via scipy.optimize.milp):
    min  Hamming(x, x_hat)   s.t.  x partial injective, evidence, F(x) <= T + tol,  x in region R_c
R_c is a linear encoding of "the RECIST inputs derived from x give overall response c" (see rules.py), deliberately OVER-approximated
by one tenth of a mm at each rational boundary and by `tol` on the energy budget, so that infeasibility of every R_c is a sound proof.
Every feasible solution is re-verified exactly (energy in float, response by the repository engine); a solution that fails
re-verification is excluded by a no-good cut and the problem is re-solved. Pairs with d_ij - g > q + tol are pruned: any matching that
contains such a pair has F(x) >= (d_ij - g) + F*_0 > T (proof in docs/proof_soundness.md).
Statuses: PROVED_EQUIVALENT | COUNTEREXAMPLE_FOUND | INCONCLUSIVE_TIMEOUT | INCONCLUSIVE_INCOMPLETE_SET | INVALID_CLINICAL_INPUT."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from . import schema as S
from .energy import Prepared, best
from .rules import MEASURABLE_TENTHS, targets_of, thresholds


def _regions(P: Prepared, tg):
    """Response classes to test for this patient (NE is only reachable when there is no target lesion)."""
    return ['PD_new', 'PD_target', 'SD', 'PR', 'CR'] if tg else ['PD_new', 'NE']


def _build(P, pairs_ok, region, x_hat, T, tol, forced, no_partner, cuts):
    tg = targets_of(P.bl_t); tgs = set(tg)
    s0t = sum(P.bl_t[a] for a in tg)
    pd, pr = thresholds(s0t)
    big = [f for f in P.fids if P.fu_t[f] >= MEASURABLE_TENTHS]
    nx = len(pairs_ok)
    use_u = region == 'PD_new'
    nu = len(big) if use_u else 0
    nv = nx + nu
    hat = set(x_hat)
    c = np.array([-1.0 if p in hat else 1.0 for p in pairs_ok] + [0.0] * nu)
    lb, ub = np.zeros(nv), np.ones(nv)
    forced_f = {f: b for b, f in forced}
    forced_b = {b: f for b, f in forced}
    for k, (b, f) in enumerate(pairs_ok):
        if f in no_partner or (f in forced_f and forced_f[f] != b) or (b in forced_b and forced_b[b] != f):
            ub[k] = 0
        if forced_f.get(f) == b:
            lb[k] = 1
    rows, lo, hi = [], [], []

    def add(coef: dict, l, h):
        rows.append(coef); lo.append(l); hi.append(h)

    byb, byf = {}, {}
    for k, (b, f) in enumerate(pairs_ok):
        byb.setdefault(b, []).append(k); byf.setdefault(f, []).append(k)
    for ks in list(byb.values()) + list(byf.values()):
        if len(ks) > 1:
            add({k: 1 for k in ks}, -np.inf, 1)
    bi, fi = {b: i for i, b in enumerate(P.bids)}, {f: j for j, f in enumerate(P.fids)}
    add({k: P.d[bi[b], fi[f]] - P.g for k, (b, f) in enumerate(pairs_ok)}, -np.inf, T + tol)
    s1 = {k: P.fu_t[f] for k, (b, f) in enumerate(pairs_ok) if b in tgs}
    if region == 'PD_new':
        for u, f in enumerate(big):                      # u_f <= 1 - sum_i x_if ; sum u >= 1
            add({nx + u: 1, **{k: 1 for k in byf.get(f, [])}}, -np.inf, 1)
        if not big:
            return None
        add({nx + u: 1 for u in range(nu)}, 1, np.inf)
    elif region in ('PD_target', 'SD', 'PR', 'CR', 'NE'):
        if region != 'PD_target':
            for f in big:                                # no new lesion: every follow-up lesion >= 10 mm is matched
                ks = byf.get(f, [])
                if not ks:
                    return None
                add({k: 1 for k in ks}, 1, np.inf)
        if region == 'PD_target':
            add(s1, pd - 1, np.inf)                      # over-approximation by 0.1 mm
        elif region == 'SD':
            add(s1, pr, pd)                              # true SD: pr+1 <= S1 <= pd-1
        elif region == 'PR':
            add(s1, -np.inf, pr + 1)
            add({k: 1 for k in range(nx)}, 1, np.inf)    # at least one baseline lesion persists (else overall CR)
        elif region == 'CR':
            add({k: 1 for k in range(nx)}, -np.inf, 0)
    for cut in cuts:                                     # no-good cuts: exclude re-verification failures
        on = [k for k, p in enumerate(pairs_ok) if p in cut]
        add({**{k: 1 for k in on}, **{k: -1 for k in range(nx) if k not in on}}, -np.inf, len(on) - 1)
    if nv == 0:                                          # no candidate pair: the empty matching is the only hypothesis
        c, lb, ub, nv = np.zeros(1), np.zeros(1), np.zeros(1), 1
    A = lil_matrix((len(rows), nv))
    for r, coef in enumerate(rows):
        for k, v in coef.items():
            A[r, k] = v
    integrality = np.ones(nv)
    return c, LinearConstraint(A.tocsr(), np.array(lo), np.array(hi)), Bounds(lb, ub), integrality, nx


def verify(P: Prepared, q: float, F0: float, forced=(), no_partner=(), time_limit=30.0, tol=1e-6, max_cuts=5) -> S.DecisionCheck:
    """F0: unconstrained minimum energy (the conformal set is anchored there, so evidence can only shrink it)."""
    t0 = time.perf_counter()
    if not P.valid:
        return S.DecisionCheck(P.pid, S.INVALID_CLINICAL_INPUT, 'NE', q, solver_log=[P.invalid_reason])
    x_hat, Fh = best(P, forced, no_partner)
    T = F0 + q
    log = []
    if x_hat is None or Fh > T + tol:
        # every hypothesis consistent with the revealed (reference) evidence lies outside the conformal set: set mis-coverage detected
        y0 = P.response(x_hat or [])[0] if x_hat is not None else 'NE'
        return S.DecisionCheck(P.pid, S.INCONCLUSIVE_INCOMPLETE_SET, y0, q, solver_seconds=time.perf_counter() - t0,
                               solver_log=[f'evidence-consistent minimum F={Fh:.2f} > T={T:.2f}'])
    y0 = P.response(x_hat)[0]
    pairs_ok = [(b, f) for i, b in enumerate(P.bids) for j, f in enumerate(P.fids) if P.d[i, j] - P.g <= q + tol]
    pairs_ok += [p for p in forced if p not in pairs_ok]
    pairs_ok += [p for p in x_hat if p not in pairs_ok]
    reach, cands, timeout = {y0}, [], False
    for region in _regions(P, targets_of(P.bl_t)):
        target_cls = 'PD' if region.startswith('PD') else region
        if target_cls == y0:
            continue
        cuts = []
        while True:
            built = _build(P, pairs_ok, region, x_hat, T, tol, forced, set(no_partner), cuts)
            if built is None:
                log.append(f'{region}: structurally infeasible'); break
            c, cons, bnd, integ, nx = built
            remaining = max(1.0, time_limit - (time.perf_counter() - t0))
            res = milp(c, constraints=cons, bounds=bnd, integrality=integ, options=dict(time_limit=remaining, presolve=True))
            if res.status == 2:
                log.append(f'{region}: infeasible (proof)'); break
            if res.x is None:
                log.append(f'{region}: no solution, status {res.status} ({res.message})'); timeout = True; break
            x = sorted(p for k, p in enumerate(pairs_ok) if res.x[k] > .5)
            r, inp, _ = P.response(x)
            if r != y0 and P.F(x) <= T + tol:
                reach.add(r)
                cands.append((len(set(x) ^ set(x_hat)), P.F(x) - Fh, x, r, inp, res.status == 0, region))
                log.append(f'{region}: counterexample -> {r}, hamming {len(set(x) ^ set(x_hat))}, optimal={res.status == 0}')
                break
            cuts.append(set(x))
            log.append(f'{region}: re-verification failed (engine {r}), no-good cut {len(cuts)}')
            if len(cuts) > max_cuts:
                timeout = True; log.append(f'{region}: cut limit'); break
    dt = time.perf_counter() - t0
    if cands:
        h, gap, x, r, inp, opt, region = min(cands, key=lambda z: (z[0], z[1]))
        rem = sorted(set(x_hat) - set(x)); add = sorted(set(x) - set(x_hat))
        les = sorted({a for a, _ in rem + add} | {b for _, b in rem + add})
        cex = S.CandidateState(S.MatchHypothesis(tuple(x), P.F(x), f'milp_{region}'), r,
                               {k: v for k, v in inp.items() if k != 'targets'})
        return S.DecisionCheck(P.pid, S.COUNTEREXAMPLE_FOUND, y0, q, sorted(reach), cex, rem, add, les, gap, dt,
                               log + [f'witness minimal={opt}'])
    st = S.INCONCLUSIVE_TIMEOUT if timeout else S.PROVED_EQUIVALENT
    return S.DecisionCheck(P.pid, st, y0, q, sorted(reach), solver_seconds=dt, solver_log=log)
