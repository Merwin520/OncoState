"""Experiment 2.3 rule layer.

spec_response   : independent RECIST 1.1 implementation written from the guideline text (exact arithmetic, mm on a 0.1 grid).
                  It defines the reference response for every case.
engine_response : the repository's deterministic rule engine (app/service/.../recist_service.py) called through an evidence
                  validator that abstains when required evidence is missing; returns a full provenance trace.
The two are compared on every generated case; any disagreement is reported, never hidden.
"""
from __future__ import annotations
import importlib.util
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_ENGINE_PATH = REPO / 'app/service/app/services/oncology/recist_service.py'
_spec = importlib.util.spec_from_file_location('recist_service', _ENGINE_PATH)
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
ENGINE = _mod.RecistService()
RESPONSES = ['CR', 'PR', 'SD', 'PD', 'NE']
NONTARGET = ['CR', 'ABSENT', 'NON-CR/NON-PD', 'PD']

# rule-id vocabulary (one id per guideline clause used)
RULE_TEXT = {
    'R.T.CR': 'Target CR: disappearance of all target lesions; pathologic nodes <10 mm short axis (RECIST 1.1)',
    'R.T.PR': 'Target PR: >=30% decrease in sum of diameters, baseline sum as reference (RECIST 1.1)',
    'R.T.PD': 'Target PD: >=20% increase AND >=5 mm absolute increase, smallest sum on study (nadir) as reference (RECIST 1.1)',
    'R.T.SD': 'Target SD: neither sufficient shrinkage for PR nor sufficient increase for PD (RECIST 1.1)',
    'R.O.NEW': 'Overall PD: unequivocal new lesion (RECIST 1.1)',
    'R.O.NTPD': 'Overall PD: unequivocal progression of non-target disease (RECIST 1.1)',
    'R.O.TPD': 'Overall PD: target-lesion PD (RECIST 1.1)',
    'R.O.CR': 'Overall CR: target CR, non-target CR, no new lesions (RECIST 1.1)',
    'R.O.PR': 'Overall PR: target CR/PR, non-target not PD, no new lesions (RECIST 1.1)',
    'R.O.SD': 'Overall SD: target SD, non-target not PD, no new lesions (RECIST 1.1)',
    'POLICY.ABSTAIN': 'Required evidence missing: no response category is assigned; manual review required (OncoAgent policy)',
}
ENGINE_RULE_TO_ID = {
    'All target lesions disappeared and any pathologic lymph nodes are <10 mm short axis.': 'R.T.CR',
    'At least 20% increase from nadir with at least 5 mm absolute increase. PD overrides shrinkage from baseline.': 'R.T.PD',
    'At least 30% decrease in target-lesion sum relative to baseline without PD from nadir.': 'R.T.PR',
    'Neither PR nor PD criteria are met.': 'R.T.SD',
    'New lesion(s) explicitly present.': 'R.O.NEW',
    'Unequivocal progression of non-target disease explicitly recorded.': 'R.O.NTPD',
    'Target-lesion PD criteria met.': 'R.O.TPD',
    'Target and non-target disease both meet CR with no new lesions.': 'R.O.CR',
    'Target response is CR/PR, non-target disease is not PD, and no new lesions are present.': 'R.O.PR',
    'Target disease is SD, non-target disease is not PD, and no new lesions are present.': 'R.O.SD',
}


def _F(x):
    return None if x is None else Fraction(str(x))


def spec_target(history, current, all_disappeared=False, nodes_lt10=False):
    """Target-lesion response and rule id. history[0] is the baseline sum; history includes all earlier sums."""
    base = _F(history[0]); cur = _F(current); nadir = min(_F(h) for h in history)
    if cur == 0 and all_disappeared and nodes_lt10:
        return 'CR', 'R.T.CR'
    if nadir > 0 and cur * 5 >= nadir * 6 and cur - nadir >= 5:          # >=20% (exact) and >=5 mm
        return 'PD', 'R.T.PD'
    if base > 0 and cur * 10 <= base * 7:                                  # <=70% of baseline (exact)
        return 'PR', 'R.T.PR'
    return 'SD', 'R.T.SD'


def spec_response(case):
    """Reference overall response, rule ids, and whether the case is an abstention case."""
    hist, cur = case.get('history'), case.get('current_sum')
    new, nt = case.get('new_lesions'), case.get('non_target')
    if new is True:                                                        # a new lesion is decisive on its own
        return 'PD', ['R.O.NEW'], False
    if hist is None or cur is None or len(hist) == 0 or case.get('lesions_missing'):
        return 'NE', ['POLICY.ABSTAIN'], True
    t, tid = spec_target(hist, cur, case.get('all_disappeared', False), case.get('nodes_lt10', False))
    if nt == 'PD':
        return 'PD', ['R.O.NTPD'], False
    if t == 'PD':
        return 'PD', [tid, 'R.O.TPD'], False
    if new is None or nt is None:
        return 'NE', [tid, 'POLICY.ABSTAIN'], True
    if t == 'CR' and nt in ('CR', 'ABSENT'):
        return 'CR', [tid, 'R.O.CR'], False
    if t in ('CR', 'PR') and nt in ('CR', 'ABSENT', 'NON-CR/NON-PD'):
        return 'PR', [tid, 'R.O.PR'], False
    if t == 'SD' and nt in ('CR', 'ABSENT', 'NON-CR/NON-PD'):
        return 'SD', [tid, 'R.O.SD'], False
    return 'NE', [tid, 'POLICY.ABSTAIN'], True


def derived(case):
    hist, cur = case.get('history'), case.get('current_sum')
    if not hist or cur is None:
        return {}
    b, n = float(hist[0]), float(min(hist))
    return {'baseline_sum': b, 'nadir_sum': n, 'current_sum': float(cur),
            'change_baseline_pct': (float(cur) - b) / b * 100 if b else None,
            'change_nadir_pct': (float(cur) - n) / n * 100 if n else None,
            'absolute_change_from_nadir_mm': float(cur) - n}


def engine_response(case):
    """OncoAgent rule module: evidence validation -> repository engine -> provenance trace."""
    trace = {'inputs': {k: case.get(k) for k in ('history', 'current_sum', 'new_lesions', 'non_target', 'all_disappeared', 'nodes_lt10')}, 'steps': []}
    hist, cur = case.get('history'), case.get('current_sum')
    new = case.get('new_lesions')
    if new is True:                                                        # new lesion decides without needing measurements
        trace['steps'].append({'step': 'new_lesion', 'rule_id': 'R.O.NEW'})
        return {'response': 'PD', 'rule_ids': ['R.O.NEW'], 'abstained': False, 'trace': trace, 'target_response': None}
    if hist is None or cur is None or len(hist) == 0 or case.get('lesions_missing'):
        trace['steps'].append({'step': 'validate', 'result': 'required measurement missing', 'rule_id': 'POLICY.ABSTAIN'})
        return {'response': 'NE', 'rule_ids': ['POLICY.ABSTAIN'], 'abstained': True, 'trace': trace, 'target_response': None}
    nt = case.get('non_target')
    res = ENGINE.evaluate(baseline_sum_mm=float(hist[0]), current_sum_mm=float(cur), nadir_sum_mm=float(min(hist)),
                          all_target_lesions_disappeared=bool(case.get('all_disappeared', False)),
                          all_pathologic_nodes_lt10mm=bool(case.get('nodes_lt10', False)),
                          non_target_response=nt if nt else 'NE', new_lesions=new)
    tid = ENGINE_RULE_TO_ID.get(res['target_rule'], 'R.T.?')
    oid = ENGINE_RULE_TO_ID.get(res['overall_rule'])
    trace['steps'] += [{'step': 'derive', **derived(case)}, {'step': 'target_rule', 'rule_id': tid, 'target_response': res['target_response']}]
    resp = res['overall_response']
    if oid is None:                                                        # engine text for the NE branches
        oid = 'POLICY.ABSTAIN'
    trace['steps'].append({'step': 'overall_rule', 'rule_id': oid, 'overall_response': resp})
    ids = [tid, oid] if oid != 'R.O.NEW' else [oid]
    if oid == 'R.O.NTPD':
        ids = [oid]
    return {'response': resp, 'rule_ids': ids, 'abstained': resp == 'NE', 'trace': trace, 'target_response': res['target_response']}
