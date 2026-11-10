"""Clinical decision function h(x): lesion correspondences -> RECIST 1.1 overall response.

The response is ALWAYS computed by the repository rule engine (app/service/.../recist_service.py) through the Exp 2.3 validated wrapper
`engine_response` (new lesion -> PD before measurements; missing target history -> NE abstention). This module only derives the
engine inputs from a correspondence hypothesis; it never decides a response itself. Input derivation follows the Exp 3.0/3.3
downstream-assessment convention (outputs/exp33 rules.assess), extended with non-target status because the engine needs it:
  targets      : <=5 largest baseline lesions with diameter >= 10 mm (organ unknown -> no per-organ cap; documented assumption)
  S0 / S1      : baseline target sum / sum of matched follow-up partner diameters (0 for an unmatched target)
  new lesion   : any follow-up lesion >= 10 mm without a baseline partner
  non-target   : ABSENT if no non-target baseline lesion; CR if none of them has a partner; otherwise NON-CR/NON-PD
                 (unequivocal non-target progression cannot be derived from diameters -> never PD; documented assumption)
  nodes < 10mm : True whenever all target lesions disappeared (nodal status unknown; documented assumption)
  nadir        : baseline (two-visit series)
All diameters are the displayed 0.1 mm values (integers in tenths of a mm)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/exp2_guideline_reasoning'))
from exp23_rules import engine_response, spec_response  # noqa: E402  (repository engine + independent spec implementation)

MEASURABLE_TENTHS = 100          # 10.0 mm
MAX_TARGETS = 5
RESPONSES = ('CR', 'PR', 'SD', 'PD', 'NE')


def tenths(x):
    return int(round(float(x) * 10))


def targets_of(bl_tenths: dict):
    return [a for a, d in sorted(bl_tenths.items(), key=lambda kv: (-kv[1], kv[0])) if d >= MEASURABLE_TENTHS][:MAX_TARGETS]


def thresholds(s0t: int):
    """Exact integer (tenths) boundaries of the rational RECIST rule: PD iff S1 >= pd; PR iff S1 <= pr."""
    pd = max(-(-6 * s0t // 5), s0t + 50)        # S1 >= 1.2*S0 and S1 - S0 >= 5 mm
    pr = (7 * s0t) // 10                         # S1 <= 0.7*S0
    return pd, pr


def rule_inputs(bl_t: dict, fu_t: dict, pairs) -> dict:
    tg = targets_of(bl_t)
    partner = {a: b for a, b in pairs}
    matched_fu = {b for _, b in pairs}
    s0t = sum(bl_t[a] for a in tg)
    s1t = sum(fu_t[partner[a]] for a in tg if a in partner)
    new = any(d >= MEASURABLE_TENTHS for b, d in fu_t.items() if b not in matched_fu)
    nts = [a for a in bl_t if a not in tg]
    nt = 'ABSENT' if not nts else ('CR' if all(a not in partner for a in nts) else 'NON-CR/NON-PD')
    gone = bool(tg) and all(a not in partner for a in tg)
    return dict(history=[s0t / 10] if tg else [], current_sum=s1t / 10 if tg else None, new_lesions=bool(new), non_target=nt,
                all_disappeared=gone, nodes_lt10=gone, s0_tenths=s0t, s1_tenths=s1t, targets=tg)


def response(bl_t: dict, fu_t: dict, pairs):
    """(overall response from the repository engine, engine inputs, independent-spec response)."""
    inp = rule_inputs(bl_t, fu_t, pairs)
    case = {k: inp[k] for k in ('history', 'current_sum', 'new_lesions', 'non_target', 'all_disappeared', 'nodes_lt10')}
    eng = engine_response(case)
    spec = spec_response(case)[0]
    return eng['response'], inp, spec
