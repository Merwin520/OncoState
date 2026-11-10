"""Frozen longitudinal query set (Q1-Q8), deterministic reference answers and scoring. Pre-registered in experiment_config.yaml.

All reference answers are computed from the frozen reference trajectory (released-mask burden in mL, 2 decimals) of the visits that are
AVAILABLE at the query time, in chronological order. Rules (also given verbatim to the LLM agents in their system prompt):
  step direction j (j>=2): diff = B_j - B_(j-1); thr = max(0.10 * B_(j-1), 1.0 mL); increase if diff > thr, decrease if diff < -thr, else stable
  reversal at t: step t is not stable AND the most recent earlier non-stable step exists AND has the opposite direction
  nadir: minimum burden over all available visits up to and including the current one (ties -> earliest visit)
  pct change = 100 * (B_now - B_ref) / B_ref (null if B_ref == 0)
"""
import json

STEP_REL, STEP_ABS_ML = 0.10, 1.0
TOL_ML_ABS, TOL_ML_REL, TOL_PCT_ABS = 0.1, 0.01, 1.0
QIDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
LONGITUDINAL = ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']      # require information from earlier visits
HISTORY_RETRIEVAL = ['Q2', 'Q8']                               # retrieve earlier measurements
TREATMENTS = ['surgery', 'radiation', 'initial_chemotherapy', 'additional_therapy', 'second_additional_therapy', 'immunotherapy',
              'brachytherapy', 'other_therapy']

TEXT = {
    'Q1': ('What is the current quantitative disease burden (burden_ml) at the current visit?',
           '{"visit": "<visit id>", "burden_ml": <number>}'),
    'Q2': ('What was the disease burden at the immediately previous observed visit (the latest available visit before the current one)?',
           '{"previous_visit": "<visit id>", "previous_burden_ml": <number>}'),
    'Q3': ('What is the change in disease burden from the immediately previous observed visit to the current visit?',
           '{"delta_ml": <number>, "pct_change": <number or null>}'),
    'Q4': ("What is the change in disease burden from the patient's first observed visit to the current visit?",
           '{"first_visit": "<visit id>", "first_burden_ml": <number>, "delta_ml": <number>, "pct_change": <number or null>}'),
    'Q5': ('What is the historical nadir of disease burden up to and including the current visit, and how does the current burden compare with it?',
           '{"nadir_visit": "<visit id>", "nadir_burden_ml": <number>, "current_minus_nadir_ml": <number>}'),
    'Q6': ('What is the trajectory pattern at the current visit according to the step-direction and reversal rules?',
           '{"current_step": "increase" | "decrease" | "stable", "reversal": true | false}'),
    'Q7': ('Which treatment categories had started by the date of the current visit (all treatment events reported up to and including the current visit)?',
           '{"treatments_started": ["<category>", ...]}'),
    'Q8': ('List the disease burden at every observed visit of this patient up to and including the current visit, in chronological order.',
           '{"history": [{"visit": "<visit id>", "burden_ml": <number>}, ...]}'),
}


def step(prev, cur):
    thr = max(STEP_REL * prev, STEP_ABS_ML)
    d = cur - prev
    return 'increase' if d > thr else 'decrease' if d < -thr else 'stable'


def pct(now, ref):
    return None if ref == 0 else round(100.0 * (now - ref) / ref, 2)


def reference_answers(seq, treatments_by_visit=None, t_day_known=True):
    """seq: [(visit, burden_ml)] of the AVAILABLE visits in chronological order, last = current visit.
    treatments_by_visit: {visit: list|None} interval treatment reports. Returns {qid: answer} for every evaluable query."""
    cur_v, cur = seq[-1]
    ans = {'Q1': {'visit': cur_v, 'burden_ml': cur}}
    if len(seq) >= 2:
        pv, pb = seq[-2]
        fv, fb = seq[0]
        nad_v, nad_b = min(seq, key=lambda x: x[1])            # min() keeps the earliest on ties
        steps = [step(seq[i - 1][1], seq[i][1]) for i in range(1, len(seq))]
        prior = [s for s in steps[:-1] if s != 'stable']
        rev = steps[-1] != 'stable' and bool(prior) and prior[-1] != steps[-1]
        ans.update({'Q2': {'previous_visit': pv, 'previous_burden_ml': pb},
                    'Q3': {'delta_ml': round(cur - pb, 2), 'pct_change': pct(cur, pb)},
                    'Q4': {'first_visit': fv, 'first_burden_ml': fb, 'delta_ml': round(cur - fb, 2), 'pct_change': pct(cur, fb)},
                    'Q5': {'nadir_visit': nad_v, 'nadir_burden_ml': nad_b, 'current_minus_nadir_ml': round(cur - nad_b, 2)},
                    'Q6': {'current_step': steps[-1], 'reversal': rev},
                    'Q8': {'history': [{'visit': v, 'burden_ml': b} for v, b in seq]}})
    # Q7 evaluable iff the current visit's date is known; the answer is the union of the treatment reports of the AVAILABLE visits
    if treatments_by_visit is not None and treatments_by_visit.get(cur_v) is not None:
        ans['Q7'] = {'treatments_started': sorted({e['treatment'] for v, _ in seq for e in (treatments_by_visit.get(v) or [])})}
    return ans


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def close_ml(p, r):
    return _num(p) and abs(p - r) <= max(TOL_ML_ABS, TOL_ML_REL * abs(r))


def close_pct(p, r):
    if r is None:
        return p is None
    return _num(p) and abs(p - r) <= TOL_PCT_ABS


def score(qid, pred, ref):
    """True/False: is the structured answer correct? pred None (failed / invalid episode) is incorrect."""
    if pred is None:
        return False
    try:
        if qid == 'Q1':
            return str(pred['visit']) == ref['visit'] and close_ml(pred['burden_ml'], ref['burden_ml'])
        if qid == 'Q2':
            return str(pred['previous_visit']) == ref['previous_visit'] and close_ml(pred['previous_burden_ml'], ref['previous_burden_ml'])
        if qid == 'Q3':
            return close_ml(pred['delta_ml'], ref['delta_ml']) and close_pct(pred['pct_change'], ref['pct_change'])
        if qid == 'Q4':
            return (str(pred['first_visit']) == ref['first_visit'] and close_ml(pred['first_burden_ml'], ref['first_burden_ml'])
                    and close_ml(pred['delta_ml'], ref['delta_ml']) and close_pct(pred['pct_change'], ref['pct_change']))
        if qid == 'Q5':
            return (str(pred['nadir_visit']) == ref['nadir_visit'] and close_ml(pred['nadir_burden_ml'], ref['nadir_burden_ml'])
                    and close_ml(pred['current_minus_nadir_ml'], ref['current_minus_nadir_ml']))
        if qid == 'Q6':
            return pred['current_step'] == ref['current_step'] and pred['reversal'] is ref['reversal']
        if qid == 'Q7':
            return sorted(set(pred['treatments_started'])) == ref['treatments_started']
        if qid == 'Q8':
            h, rh = pred['history'], ref['history']
            return len(h) == len(rh) and all(str(a['visit']) == b['visit'] and close_ml(a['burden_ml'], b['burden_ml']) for a, b in zip(h, rh))
    except (KeyError, TypeError, AttributeError):
        return False
    return False


def validate(qid, j):
    """Format validation used by the agents' harness (exp22b_agent.validate is replaced by this; same role)."""
    try:
        if qid == 'Q1':
            return isinstance(j['visit'], str) and _num(j['burden_ml'])
        if qid == 'Q2':
            return isinstance(j['previous_visit'], str) and _num(j['previous_burden_ml'])
        if qid == 'Q3':
            return _num(j['delta_ml']) and (j['pct_change'] is None or _num(j['pct_change']))
        if qid == 'Q4':
            return isinstance(j['first_visit'], str) and _num(j['first_burden_ml']) and _num(j['delta_ml']) and (j['pct_change'] is None or _num(j['pct_change']))
        if qid == 'Q5':
            return isinstance(j['nadir_visit'], str) and _num(j['nadir_burden_ml']) and _num(j['current_minus_nadir_ml'])
        if qid == 'Q6':
            return j['current_step'] in ('increase', 'decrease', 'stable') and isinstance(j['reversal'], bool)
        if qid == 'Q7':
            return isinstance(j['treatments_started'], list) and all(x in TREATMENTS for x in j['treatments_started'])
        if qid == 'Q8':
            return isinstance(j['history'], list) and all(isinstance(x['visit'], str) and _num(x['burden_ml']) for x in j['history'])
    except (KeyError, TypeError):
        return False
    return False


def dumps(x):
    return json.dumps(x, sort_keys=True)
