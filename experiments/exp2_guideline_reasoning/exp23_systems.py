"""Experiment 2.3 systems: Full OncoAgent, No-retrieval ablation, LLM-only, Rule executor only -> benchmark_predictions_long.csv

Full        : structured state -> evidence validator -> repository RECIST engine -> provenance trace -> retrieved guideline citations
No-retrieval: identical to Full without any guideline retrieval (no citations)
LLM-only    : gpt-5.2 sees the same evidence, may not call the engine (5 independent repeats, raw outputs in llm_raw.jsonl)
Rule-only   : the repository engine called directly on the raw fields (no evidence validator, no provenance, no citations)
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

import exp23_rules as RL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
SYSTEMS = ['Full OncoAgent', 'No-retrieval', 'LLM-only', 'Rule executor only']


def to_case(r):
    def tri(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ('None', 'nan', '') else (str(v) == 'True')
    hist = json.loads(r.history_json) if isinstance(r.history_json, str) and r.history_json not in ('null', '') else None
    nt = None if (r.non_target_status is None or (isinstance(r.non_target_status, float) and np.isnan(r.non_target_status))) else str(r.non_target_status)
    return dict(history=hist, current_sum=None if pd.isna(r.current_sum) else float(r.current_sum), new_lesions=tri(r.new_lesion), non_target=nt,
                all_disappeared=bool(r.all_disappeared), nodes_lt10=bool(r.nodes_lt10), lesions_missing=bool(r.lesions_missing))


def ids_from_engine(res):
    tid = RL.ENGINE_RULE_TO_ID.get(res['target_rule'], 'R.T.?'); oid = RL.ENGINE_RULE_TO_ID.get(res['overall_rule'], 'POLICY.ABSTAIN')
    return [oid] if oid in ('R.O.NEW', 'R.O.NTPD') else [tid, oid]


def raw_engine(case):
    try:
        h = case['history']; res = RL.ENGINE.evaluate(baseline_sum_mm=h[0], current_sum_mm=case['current_sum'], nadir_sum_mm=min(h),
                                                       all_target_lesions_disappeared=case['all_disappeared'], all_pathologic_nodes_lt10mm=case['nodes_lt10'],
                                                       non_target_response=case['non_target'] or 'NE', new_lesions=case['new_lesions'])
        return res['overall_response'], ids_from_engine(res), True
    except Exception:
        return 'ERROR', [], False


def clause_table():
    p = OUT / 'guideline_retrieval.json'
    if not p.exists(): return {}
    return json.loads(p.read_text())['clauses']


def citation_fields(ids, clauses):
    need = [i for i in ids if not i.startswith('POLICY')]
    if not need: return None, None, None
    present = all(i in clauses and clauses[i]['citations'] and clauses[i]['text'] for i in need)
    correct = present and all(clauses[i]['citation_correct'] for i in need)
    supported = present and all(clauses[i]['citation_supported'] and clauses[i]['citation_correct'] for i in need)
    return present, correct, supported


PAT = {'R.T.PR': lambda t: '30' in t and ('baseline' in t or 'decrease' in t or 'decline' in t),
       'R.T.PD': lambda t: '20' in t and re.search(r'\b5\s*mm', t) and ('nadir' in t or 'smallest' in t),
       'R.T.SD': lambda t: any(w in t for w in ('stable', 'neither', 'not meet', 'sd')), 'R.T.CR': lambda t: 'disappear' in t,
       'R.O.NEW': lambda t: 'new lesion' in t, 'R.O.NTPD': lambda t: 'non-target' in t or 'non target' in t,
       'R.O.TPD': lambda t: 'target' in t and ('pd' in t or 'progress' in t),
       'R.O.CR': lambda t: 'non-target' in t or 'no new' in t or 'complete' in t, 'R.O.PR': lambda t: 'non-cr' in t or 'non-target' in t or 'no new' in t or 'partial' in t,
       'R.O.SD': lambda t: 'non-cr' in t or 'non-target' in t or 'no new' in t or 'stable' in t,
       'POLICY.ABSTAIN': lambda t: any(w in t for w in ('insufficient', 'missing', 'not available', 'cannot', 'unknown', 'not assessed', 'incomplete', 'not measured', 'unable'))}


def parse_llm(raw):
    try:
        j = json.loads(raw[raw.index('{'):raw.rindex('}') + 1]); resp = str(j.get('response', '')).strip().upper()
        return (resp if resp in RL.RESPONSES else 'PARSE_ERROR'), j
    except Exception:
        return 'PARSE_ERROR', {}


def build():
    cases = pd.read_csv(OUT / 'rule_challenge_cases.csv'); clauses = clause_table(); rows = []
    for r in cases.itertuples():
        c = to_case(r); exp, exp_ids = r.expected_response, r.expected_rule_ids.split(';'); ea = bool(r.expected_abstention)
        base = dict(case_id=r.case_id, subset=r.subset, challenge_type=r.challenge_type, reference_response=exp, expected_abstention=ea)
        eng = RL.engine_response(c); ids = eng['rule_ids']
        pres, corr, sup = citation_fields(ids, clauses)
        chain_ok = bool(eng['trace']['steps']) and all('rule_id' in s for s in eng['trace']['steps'] if s['step'] != 'derive') and eng['trace']['inputs'] is not None
        common = dict(predicted_response=eng['response'], correct=eng['response'] == exp, rule_clause_correct=set(ids) == set(exp_ids), traceability_complete=chain_ok,
                      abstained=eng['response'] == 'NE', rule_ids_used=';'.join(ids), repeat_id=0, latency_sec=0.0, token_count=0, raw_answer='')
        rows.append({**base, 'system': 'Full OncoAgent', **common, 'citation_present': pres, 'citation_correct': corr, 'citation_supported': sup})
        rows.append({**base, 'system': 'No-retrieval', **common, 'citation_present': (False if pres is not None else None), 'citation_correct': (False if pres is not None else None), 'citation_supported': (False if pres is not None else None)})
        resp, rids, ok = raw_engine(c)
        rows.append({**base, 'system': 'Rule executor only', **{**common, 'predicted_response': resp, 'correct': resp == exp, 'rule_clause_correct': ok and set(rids) == set(exp_ids),
                     'traceability_complete': ok, 'abstained': resp == 'NE', 'rule_ids_used': ';'.join(rids)}, 'citation_present': (False if pres is not None else None), 'citation_correct': (False if pres is not None else None), 'citation_supported': (False if pres is not None else None)})
        true = RL.derived(c)
        raw = OUT / 'llm_raw.jsonl'
    if (OUT / 'llm_raw.jsonl').exists():
        meta = {r.case_id: r for r in cases.itertuples()}
        truth = {cid: RL.derived(to_case(meta[cid])) for cid in meta}
        for l in (OUT / 'llm_raw.jsonl').read_text().splitlines():
            j = json.loads(l)
            if j.get('error'): continue
            r = meta[j['case_id']]; exp, exp_ids = r.expected_response, r.expected_rule_ids.split(';'); pred, obj = parse_llm(j['raw_answer']); t = truth[j['case_id']]
            txt = str(obj.get('rule_applied', '')).lower()
            rule_ok = bool(txt) and all(bool(PAT.get(i, lambda s: False)(txt)) for i in exp_ids)
            def close(a, b):
                return (a is None and b is None) or (a is not None and b is not None and isinstance(a, (int, float)) and abs(float(a) - float(b)) <= 0.2)
            cb, cn = obj.get('computed_change_from_baseline_pct'), obj.get('computed_change_from_nadir_pct')
            tb, tn = t.get('change_baseline_pct'), t.get('change_nadir_pct')
            trace_ok = pred != 'PARSE_ERROR' and bool(txt) and (close(cb, tb) or (tb is None and cb is None)) and (close(cn, tn) or (tn is None and cn is None))
            rows.append(dict(case_id=j['case_id'], subset=r.subset, challenge_type=r.challenge_type, reference_response=exp, expected_abstention=bool(r.expected_abstention), system='LLM-only',
                             repeat_id=j['repeat_id'], predicted_response=pred, correct=pred == exp, citation_present=('recist' in txt) if any(not i.startswith('POLICY') for i in exp_ids) else None,
                             citation_correct=None, citation_supported=(('recist' in txt) and rule_ok) if any(not i.startswith('POLICY') for i in exp_ids) else None,
                             rule_clause_correct=rule_ok, traceability_complete=trace_ok, abstained=pred == 'NE', rule_ids_used='', raw_answer=j['raw_answer'],
                             latency_sec=j['latency_sec'], token_count=j['token_count']))
    df = pd.DataFrame(rows)
    df['unsupported_decision'] = df.expected_abstention & df.predicted_response.isin(['CR', 'PR', 'SD', 'PD'])
    cols = ['case_id', 'system', 'repeat_id', 'subset', 'challenge_type', 'predicted_response', 'reference_response', 'correct', 'citation_present', 'citation_correct', 'citation_supported',
            'rule_clause_correct', 'traceability_complete', 'abstained', 'expected_abstention', 'unsupported_decision', 'rule_ids_used', 'raw_answer', 'latency_sec', 'token_count']
    df = df[cols]; df.to_csv(OUT / 'benchmark_predictions_long.csv', index=False)
    return df


if __name__ == '__main__':
    df = build()
    m = df[df.subset == 'main'].groupby('system').agg(n=('correct', 'size'), acc=('correct', 'mean'), rule=('rule_clause_correct', 'mean'), trace=('traceability_complete', 'mean'))
    print(m.round(3).to_string()); print(df.groupby('system').size().to_dict())
