"""Evidence revision and certificate re-verification (Part B). Reference-free given the event list.

A fact constrains the correspondence state: ('pair', b, f) | ('fu_none', f) | ('bl_none', b). The conformal set stays anchored at the
unconstrained minimum F*_0, so adding facts can only shrink it (C' = C ∩ facts). Hence, for a monotone addition already satisfied by
the current best state x̂, a PROVED certificate stays proved for the same response, and an INCOMPLETE_SET status stays incomplete
(x̂ remains the constrained optimum). Every other event (retraction, a fact x̂ violates, any other cached status) invalidates the
certificate. Arms:
  A1 full-history reconstruction + CDT-R   re-verify from all valid facts at every event
  A2 OncoState, dependency-aware           keep / invalidate the cached certificate as above
  A3 no certificate invalidation           state + response recomputed, certificate status of the initial verification kept
  A4 stored-value update only              facts stored; state, response and certificate not recomputed
  A5 no certification                      state + response recomputed; every decision accepted"""
from __future__ import annotations

import copy
import re
import time

import numpy as np

from .common import S, best, verify

BIG = 1e6
CONFLICT = 'EVIDENCE_CONFLICT'


def _nat(x):
    return int(re.sub(r'\D', '', x) or 0)


def facts_from_pairs(P, pairs):
    """All facts implied by a full correspondence set; baseline facts by id, then follow-up-only facts by id."""
    by_b = {}
    for b, f in pairs:
        by_b.setdefault(b, []).append(f)
    out = []
    for b in sorted(P.bids, key=_nat):
        if b in by_b:
            out += [('pair', b, f) for f in sorted(by_b[b], key=_nat)]
        else:
            out.append(('bl_none', b))
    matched_f = {f for _, f in pairs}
    out += [('fu_none', f) for f in sorted(P.fids, key=_nat) if f not in matched_f]
    return out


def consistent(facts):
    """(ok, conflicting fact or None): injective pairs; no 'none' fact on a lesion that has a pair fact."""
    fb, ff, bn, fn = {}, {}, set(), set()
    for x in facts:
        if x[0] == 'pair':
            _, b, f = x
            if fb.get(b, f) != f or ff.get(f, b) != b or b in bn or f in fn:
                return False, x
            fb[b] = f; ff[f] = b
        elif x[0] == 'bl_none':
            if x[1] in fb:
                return False, x
            bn.add(x[1])
        else:
            if x[1] in ff:
                return False, x
            fn.add(x[1])
    return True, None


def constrained(P, facts):
    forced = tuple(sorted((x[1], x[2]) for x in facts if x[0] == 'pair'))
    no_partner = tuple(sorted(x[1] for x in facts if x[0] == 'fu_none'))
    bl_none = [x[1] for x in facts if x[0] == 'bl_none']
    if bl_none:
        P = copy.copy(P); P.d = P.d.copy()
        bi = {b: i for i, b in enumerate(P.bids)}
        for b in bl_none:
            P.d[bi[b], :] = BIG
    return P, forced, no_partner


def satisfies(x_hat, fact):
    xs = set(x_hat); fs = {f for _, f in x_hat}; bs = {b for b, _ in x_hat}
    if fact[0] == 'pair':
        return (fact[1], fact[2]) in xs
    if fact[0] == 'fu_none':
        return fact[1] not in fs
    return fact[1] not in bs


class Checker:
    """Fresh (re)verification under a fact set; counts verifier calls."""

    def __init__(self, P, F0, q, vc):
        self.P, self.F0, self.q, self.vc = P, F0, q, vc
        self.calls, self.seconds = 0, 0.0

    def state(self, facts):
        ok, bad = consistent(facts)
        good = facts if ok else [x for x in facts if x != bad]
        P2, fo, npf = constrained(self.P, good)
        x, _ = best(P2, fo, npf)
        x = x or []
        return ok, x, P2, fo, npf

    def fresh(self, facts):
        ok, x, P2, fo, npf = self.state(facts)
        if not ok:
            return dict(status=CONFLICT, response=self.P.response(x)[0], x=x)
        t = time.perf_counter()
        c = verify(P2, self.q, self.F0, forced=fo, no_partner=npf, time_limit=self.vc['time_limit_s'], tol=self.vc['energy_tol'],
                   max_cuts=self.vc['max_nogood_cuts'])
        self.calls += 1; self.seconds += time.perf_counter() - t
        return dict(status=c.status, response=self.P.response(x)[0] if c.status != S.INVALID_CLINICAL_INPUT else 'NE', x=x)

    def reanchored(self, facts):
        """B5: re-anchor the set at the constrained optimum and re-verify (exploratory)."""
        ok, x, P2, fo, npf = self.state(facts)
        if not ok:
            return CONFLICT
        Fc = P2.F(x) if x else 0.0
        c = verify(P2, self.q, Fc, forced=fo, no_partner=npf, time_limit=self.vc['time_limit_s'], tol=self.vc['energy_tol'],
                   max_cuts=self.vc['max_nogood_cuts'])
        return c.status


def replay(P, F0, q, vc, events, ref_response):
    """events: list of ('add', fact) | ('retract', fact). Returns per-event rows for arms A1-A5 (+ counters)."""
    chk = Checker(P, F0, q, vc)
    facts = []
    f0 = chk.fresh(facts)
    init = dict(f0)
    a2 = dict(f0)                                   # cached certificate of A2
    a2_calls = 1
    rows = []

    def accepted_ok(status, resp, fr):              # an accepted decision is valid iff the fresh check proves the same response
        return fr['status'] == S.PROVED_EQUIVALENT and fr['response'] == resp

    for t, (kind, fact) in enumerate([('init', None)] + list(events)):
        if kind == 'add':
            facts = facts + [fact]
        elif kind == 'retract':
            facts = [x for x in facts if x != fact]
        fr = f0 if kind == 'init' else chk.fresh(facts)
        # A2: dependency-aware cache
        if kind != 'init':
            keep = (kind == 'add' and a2['status'] in (S.PROVED_EQUIVALENT, S.INCONCLUSIVE_INCOMPLETE_SET)
                    and consistent(facts)[0] and satisfies(a2['x'], fact))
            if not keep:
                a2 = dict(fr); a2_calls += 1
        # A3: recompute state/response, keep initial certificate status
        ok, x3, *_ = chk.state(facts)
        r3 = P.response(x3)[0]
        arms = {
            'A1': (fr['status'], fr['response'], fr['status'] == S.PROVED_EQUIVALENT),
            'A2': (a2['status'], a2['response'], a2['status'] == S.PROVED_EQUIVALENT),
            'A3': (init['status'], r3, init['status'] == S.PROVED_EQUIVALENT),
            'A4': (init['status'], init['response'], init['status'] == S.PROVED_EQUIVALENT),
            'A5': ('NO_CERTIFICATE', r3, True),
        }
        for arm, (st, resp, acc) in arms.items():
            rows.append(dict(step=t, kind=kind, fact=' '.join(fact) if fact else '', arm=arm, status=st, response=resp, accepted=int(acc),
                             stale_accepted=int(acc and arm != 'A5' and not accepted_ok(st, resp, fr)),
                             wrong=int(resp != ref_response), accepted_wrong=int(acc and resp != ref_response),
                             fresh_status=fr['status'], fresh_response=fr['response'],
                             a2_equals_a1=int(a2['status'] == fr['status'] and a2['response'] == fr['response'])))
    final_reanchor = chk.reanchored(facts) if fr['status'] == S.INCONCLUSIVE_INCOMPLETE_SET else ''
    return rows, dict(a1_calls=chk.calls + 1, a2_calls=a2_calls, a1_seconds=chk.seconds, final_reanchor=final_reanchor,
                      n_facts=len(facts), init_status=init['status'], init_response=init['response'])


def wrong_fact(P, x_hat):
    """R-err: pair(b2, f) for the V0-matched follow-up lesion with the smallest nearest/second-nearest baseline distance gap."""
    if P.n < 2 or not x_hat:
        return None
    bi = {b: i for i, b in enumerate(P.bids)}; fi = {f: j for j, f in enumerate(P.fids)}
    best_gap, out = np.inf, None
    for b, f in x_hat:
        col = P.d[:, fi[f]]
        o = np.argsort(col, kind='stable')
        gap = col[o[1]] - col[o[0]]
        b2 = P.bids[o[1]] if P.bids[o[0]] == b else P.bids[o[0]]
        if gap < best_gap:
            best_gap, out = gap, ('pair', b2, f)
    return out
