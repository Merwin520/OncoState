"""Decision acceptance for one patient: verifier status -> ACCEPT / VERIFY / ABSTAIN. Reference-free (run under no_reference)."""
from __future__ import annotations

import math

from .common import S, min_cal, verify

ACCEPT, VERIFY, ABSTAIN = 'ACCEPT', 'VERIFY', 'ABSTAIN'
ABSTAIN_CAL = 'ABSTAIN_INSUFFICIENT_CALIBRATION'
ACTION = {S.PROVED_EQUIVALENT: ACCEPT, S.COUNTEREXAMPLE_FOUND: VERIFY, S.INCONCLUSIVE_TIMEOUT: ABSTAIN,
          S.INCONCLUSIVE_INCOMPLETE_SET: ABSTAIN, S.INVALID_CLINICAL_INPUT: ABSTAIN, ABSTAIN_CAL: ABSTAIN}


def decide(P, F0: float, q: float, n_pool: int, alpha: float, vc: dict, enforce_pool: bool, forced=(), no_partner=()) -> dict:
    """enforce_pool: abstain when the calibration pool cannot support the conformal rank (coverage-aware rule)."""
    if enforce_pool and (n_pool < min_cal(alpha) or not math.isfinite(q)) and P.n and P.m:
        return dict(status=ABSTAIN_CAL, action=ABSTAIN, response='', reachable='', solver_s=0.0)
    c = verify(P, q, F0, forced=forced, no_partner=no_partner, time_limit=vc['time_limit_s'], tol=vc['energy_tol'],
               max_cuts=vc['max_nogood_cuts'])
    return dict(status=c.status, action=ACTION[c.status], response=c.reference_response, reachable=';'.join(c.reachable_responses),
                solver_s=c.solver_seconds)
