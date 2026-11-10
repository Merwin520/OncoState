"""OncoState CDT-R agent loop for one cohort under a GLOBAL review budget.

Per patient the agent keeps an append-only, hash-chained ledger (StateCommit): the V0 hypothesis, every evidence update and every
re-derived hypothesis/decision. Evidence is never rewritten; a new state version is derived from (V0 evidence + all evidence updates).
All policies share the same state update (constrained energy argmin -> repository RECIST engine) and differ ONLY in which follow-up
lesion they send to the (simulated) review tool next. The decision of every policy at every budget is h(x_hat).

Policies
  v0_no_review   : V0 state, no review (original OncoState matching node + engine)
  random         : uniformly random unrevealed follow-up lesion (seeded)
  v0_confidence  : lowest V0 match confidence first (matched: 1 - d/g; unmatched: clip(d_nearest/g - 1, 0, 1)); ties -> larger lesion
  set_ambiguity  : conformal-set identity uncertainty, decision-agnostic: smallest min-marginal energy gap Delta_j <= q first,
                   recomputed after every reveal
  cdtr           : decision-targeted: only patients whose decision is NOT proved equivalent; patient with the closest counterexample
                   (smallest energy gap) first; within it, the witness follow-up lesion with the smallest Delta_j (heuristic, labelled)
"""
from __future__ import annotations

import hashlib
import heapq
import json
import time

import numpy as np

from . import schema as S
from .energy import best, min_marginals, prepare
from .verifier import verify


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()


class PatientState:
    def __init__(self, ev, q, g, vcfg, prep_fn=None):
        self.P = prep_fn(ev) if prep_fn else prepare(ev, g)
        self.q, self.vcfg = q, vcfg
        self.forced, self.no_partner = [], []
        self.ledger: list[S.StateCommit] = []
        self.x0, self.F0 = best(self.P)
        assert sorted(self.x0) == sorted(self.P.v0_repository), f'{self.P.pid}: energy argmin differs from repository V0'
        self._commit('hypothesis', dict(pairs=self.x0, energy=self.F0, source='V0'))
        self.x_hat, self.F_hat = self.x0, self.F0
        self.check: S.DecisionCheck | None = None
        self.reanchored = False
        self.solver_seconds = 0.0
        self.n_verify = 0
        self._dm = None

    # ---------------------------------------------------------------- ledger
    def _commit(self, kind, payload):
        parent = self.ledger[-1].hash if self.ledger else None
        v = len(self.ledger)
        self.ledger.append(S.StateCommit(self.P.pid, v, parent, kind, payload, _h([parent, v, kind, payload])))

    def ledger_ok(self):
        prev = None
        for v, c in enumerate(self.ledger):
            if c.parent_hash != prev or c.version != v or c.hash != _h([prev, v, c.kind, c.payload]):
                return False
            prev = c.hash
        return True

    # ---------------------------------------------------------------- state update
    def observe(self, upd: S.EvidenceUpdate):
        if upd.partner_id is None:
            self.no_partner.append(upd.lesion_id)
        else:
            self.forced.append((upd.partner_id, upd.lesion_id))
        self._commit('evidence', dict(lesion_id=upd.lesion_id, partner_id=upd.partner_id, source=upd.source, step=upd.arrival_step))
        self.x_hat, self.F_hat = best(self.P, self.forced, self.no_partner)
        self._commit('hypothesis', dict(pairs=self.x_hat, energy=self.F_hat, source='constrained_argmin'))
        self.check, self._dm = None, None

    def revealed(self):
        return {f for _, f in self.forced} | set(self.no_partner)

    def decision(self):
        return self.P.response(self.x_hat)[0]

    def delta(self):
        if self._dm is None:
            self._dm = min_marginals(self.P, self.x_hat, self.forced, self.no_partner)
        return self._dm

    def verify(self):
        if self.check is None:
            t = time.perf_counter()
            c = verify(self.P, self.q, self.F0, self.forced, self.no_partner, self.vcfg['time_limit_s'], self.vcfg['energy_tol'],
                       self.vcfg['max_nogood_cuts'])
            self.reanchored = False
            if c.status == S.INCONCLUSIVE_INCOMPLETE_SET:     # revealed evidence excludes the whole set: re-anchor at the
                c2 = verify(self.P, self.q, self.F_hat, self.forced, self.no_partner,   # evidence-consistent optimum (no coverage
                            self.vcfg['time_limit_s'], self.vcfg['energy_tol'], self.vcfg['max_nogood_cuts'])  # guarantee; flagged)
                c2.solver_log = c.solver_log + ['re-anchored at evidence-consistent optimum'] + c2.solver_log
                c2.solver_seconds += c.solver_seconds
                c, self.reanchored = c2, True
            self.solver_seconds += time.perf_counter() - t
            self.n_verify += 1
            self.check = c
            self._commit('decision', dict(status=c.status, response=c.reference_response, reanchored=self.reanchored,
                                          witness=c.witness_lesions))
        return self.check

    def rebuild_from_history(self):
        """Full-history reconstruction control: rebuild the state from the original evidence + the evidence ledger only."""
        forced, nop = [], []
        for c in self.ledger:
            if c.kind == 'evidence':
                (nop.append(c.payload['lesion_id']) if c.payload['partner_id'] is None
                 else forced.append((c.payload['partner_id'], c.payload['lesion_id'])))
        x, _ = best(self.P, forced, nop)
        return x, self.P.response(x)[0]


# -------------------------------------------------------------------- acquisition policies
def v0_confidence_order(states):
    rows = []
    for p, st in states.items():
        P = st.P
        partner = {f: b for b, f in st.x0}
        bi, fi = {b: i for i, b in enumerate(P.bids)}, {f: j for j, f in enumerate(P.fids)}
        for f in P.fids:
            if f in partner:
                conf = max(0.0, 1 - P.d[bi[partner[f]], fi[f]] / P.g)
            else:
                dn = P.d[:, fi[f]].min() if P.n else np.inf
                conf = float(np.clip(dn / P.g - 1, 0, 1))
            rows.append((conf, -P.fu_t[f], p, f))
    rows.sort()
    return [(p, f) for _, _, p, f in rows]


def model_confidence_order(states):
    """Uncertainty baseline for a probabilistic matcher (P.prob[i, j] = model pair probability): matched follow-up lesion -> p of its
    pair; unmatched -> 1 - max_i p_ij. Lowest first; ties -> larger lesion."""
    rows = []
    for p, st in states.items():
        P = st.P
        partner = {f: b for b, f in st.x0}
        bi, fi = {b: i for i, b in enumerate(P.bids)}, {f: j for j, f in enumerate(P.fids)}
        for f in P.fids:
            pr = P.prob[:, fi[f]] if P.n else np.zeros(0)
            conf = float(pr[bi[partner[f]]]) if f in partner else float(1 - (pr.max() if pr.size else 0.0))
            rows.append((conf, -P.fu_t[f], p, f))
    rows.sort()
    return [(p, f) for _, _, p, f in rows]


def run_policy(policy, evidence, q, cfg, review_tool, K_max, checkpoints, seed=0, log=print, prep_fn=None):
    """Returns {K: {pid: record}} at each checkpoint K (reviews used so far) and per-policy runtime."""
    g, vcfg = cfg['energy']['gate_mm'], cfg['verifier']
    t0 = time.perf_counter()
    states = {p: PatientState(e, q, g, vcfg, prep_fn) for p, e in sorted(evidence.items())}
    out, used, step = {}, 0, 0
    cps = sorted(set(checkpoints))

    def snapshot():
        rec = {}
        for p, st in states.items():
            r = dict(decision=st.decision(), pairs=st.x_hat, n_reviews=len(st.revealed()))
            if policy in ('cdtr', 'set_ambiguity'):
                c = st.verify() if policy == 'cdtr' else None
                if c is not None:
                    r.update(status=c.status, reachable=c.reachable_responses, reanchored=st.reanchored)
                if policy == 'set_ambiguity':
                    r['identity_certain'] = not any(v <= q for v in st.delta().values())
            rec[p] = r
        return rec

    def reveal(p, f):
        nonlocal used, step
        step += 1
        st = states[p]
        st.observe(review_tool(p, f, policy, step))
        used += 1

    # ---------------- selection loops
    if policy == 'v0_no_review':
        order = []
    elif policy == 'random':
        rng = np.random.default_rng(seed)
        order = [(p, f) for p, st in states.items() for f in st.P.fids]
        order = [order[i] for i in rng.permutation(len(order))]
    elif policy == 'v0_confidence':
        order = v0_confidence_order(states)
    elif policy == 'model_confidence':
        order = model_confidence_order(states)
    else:
        order = None

    ci = 0
    if 0 in cps:
        out[0] = snapshot()
    if order is not None:
        for p, f in order[:K_max]:
            reveal(p, f)
            if used in cps:
                out[used] = snapshot()
    elif policy == 'set_ambiguity':
        heap = []
        for p, st in states.items():
            for f, dlt in st.delta().items():
                if dlt <= q:
                    heapq.heappush(heap, (dlt, -st.P.fu_t[f], p, f, len(st.ledger)))
        while used < K_max and heap:
            dlt, _, p, f, ver = heapq.heappop(heap)
            st = states[p]
            if ver != len(st.ledger):          # stale entry: patient changed since this priority was computed
                continue
            reveal(p, f)
            for f2, d2 in st.delta().items():
                if d2 <= q:
                    heapq.heappush(heap, (d2, -st.P.fu_t[f2], p, f2, len(st.ledger)))
            if used in cps:
                out[used] = snapshot()
    elif policy == 'cdtr':
        def push(heap, p):
            st = states[p]
            c = st.verify()
            if c.status == S.COUNTEREXAMPLE_FOUND:
                wf = [f for f in c.witness_lesions if f in set(st.P.fids) and f not in st.revealed()]
                if wf:
                    heapq.heappush(heap, (0, c.counterexample_gap, p))
                    return
            if c.status == S.INCONCLUSIVE_TIMEOUT and any(v <= q for v in st.delta().values()):
                heapq.heappush(heap, (1, min(st.delta().values()), p))
        heap = []
        for p in states:
            push(heap, p)
        while used < K_max and heap:
            tier, _, p = heapq.heappop(heap)
            st = states[p]
            c = st.verify()
            dm = st.delta()
            if tier == 0:
                cand = [f for f in c.witness_lesions if f in dm]
            else:
                cand = [f for f, v in dm.items() if v <= q]
            if not cand:
                continue
            f = min(cand, key=lambda z: (dm[z], -st.P.fu_t[z], z))
            reveal(p, f)
            push(heap, p)
            if used in cps:
                out[used] = snapshot()
    # budget not exhausted (policy stopped by itself): later checkpoints equal the final state
    final = snapshot() if (used not in out) else out[used]
    for K in cps:
        if K not in out and K >= used:
            out[K] = final
    runtime = time.perf_counter() - t0
    recon = sum(st.rebuild_from_history()[1] != st.decision() or sorted(st.rebuild_from_history()[0]) != sorted(st.x_hat)
                for st in states.values())
    meta = dict(policy=policy, seed=seed, reviews_used=used, runtime_s=runtime,
                solver_seconds=sum(st.solver_seconds for st in states.values()), n_verify=sum(st.n_verify for st in states.values()),
                reconstruction_mismatches=int(recon), ledger_ok=all(st.ledger_ok() for st in states.values()),
                ledger_commits=sum(len(st.ledger) for st in states.values()))
    log(json.dumps(meta))
    return out, meta, states
