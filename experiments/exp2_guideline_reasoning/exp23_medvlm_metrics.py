"""Fold the open-weight medical-VLM baselines (LLaVA-Med, MedGemma; text-only, greedy, single run/case) into the
Experiment 2.3 comparison, using the SAME metric definitions as the other systems (exp23_systems.py / exp23_stats.py):
accuracy, rule-clause correctness (from the free-text 'rule_applied' field, same regex checks as LLM-only), traceability
(numeric change re-derivation, same tolerance as LLM-only), abstention/false-abstention rates, and the 5 core classification
metrics (macro precision, macro recall, macro F1, weighted F1, Cohen's kappa) over the 5-way RECIST label.
Two evaluation sources: the synthetic 'main' rule-challenge subset (2000 cases, same set the 4 original systems are scored
on) and the real HCUCH cohort (23 cases, target-lesion response only, no rule-id ground truth -> accuracy + classification
metrics only, no rule/trace/abstention columns).
Writes: medvlm_predictions_long.csv (case-level, same columns as benchmark_predictions_long.csv where applicable),
medvlm_main_table.csv (system-level, same columns as exp23_main_table.csv).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import exp23_rules as RL
import exp23_systems as SY
import exp23_stats as ST

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
MODELS = {'llava_med': 'Medical VLM (LLaVA-Med v1.5 7B, text-only)', 'medgemma': 'Medical VLM (MedGemma 4B-it, text-only)'}


def main_subset_rows(mk, name, raw, cases):
    meta = {r.case_id: r for r in cases.itertuples()}
    truth = {cid: RL.derived(SY.to_case(meta[cid])) for cid in meta}
    rows = []
    for j in raw:
        if j['case_id'] not in meta: continue
        r = meta[j['case_id']]; exp, exp_ids = r.expected_response, r.expected_rule_ids.split(';'); t = truth[j['case_id']]
        pred = j.get('predicted_response', 'PARSE_ERROR'); obj = j.get('structured') or {}
        txt = str(obj.get('rule_applied', '')).lower()
        rule_ok = bool(txt) and all(bool(SY.PAT.get(i, lambda s: False)(txt)) for i in exp_ids)
        def close(a, b): return (a is None and b is None) or (a is not None and b is not None and isinstance(a, (int, float)) and abs(float(a) - float(b)) <= 0.2)
        cb, cn = obj.get('computed_change_from_baseline_pct'), obj.get('computed_change_from_nadir_pct'); tb, tn = t.get('change_baseline_pct'), t.get('change_nadir_pct')
        trace_ok = pred != 'PARSE_ERROR' and bool(txt) and (close(cb, tb)) and (close(cn, tn))
        rows.append(dict(case_id=j['case_id'], system=name, repeat_id=0, subset='main', challenge_type=r.challenge_type, reference_response=exp, predicted_response=pred, correct=pred == exp,
                         expected_abstention=bool(r.expected_abstention), rule_clause_correct=rule_ok, traceability_complete=trace_ok, abstained=pred == 'NE',
                         unsupported_decision=bool(r.expected_abstention) and pred in ('CR', 'PR', 'SD', 'PD'), raw_answer=j.get('raw_answer', ''), latency_sec=j.get('latency_sec'), token_count=j.get('input_tokens')))
    return pd.DataFrame(rows)


def real_rows(mk, name, raw, real):
    meta = {f'{r.patient_id}_{r.study_pair}': r for r in real.itertuples()}
    rows = []
    for j in raw:
        if j['case_id'] not in meta: continue
        r = meta[j['case_id']]; pred = j.get('predicted_response', 'PARSE_ERROR')
        rows.append(dict(case_id=j['case_id'], system=name, subset='real_hcuch', reference_response=r.reference_response, predicted_response=pred, correct=pred == r.reference_response, raw_answer=j.get('raw_answer', '')))
    return pd.DataFrame(rows)


def summarize_main(d, n_planned):
    m = ST.cls_metrics(d.reference_response, d.predicted_response); ab = d[d.expected_abstention]; nab = d[~d.expected_abstention]
    lo_acc, hi_acc = ST.wilson(int(d.correct.sum()), len(d)); lo_rule, hi_rule = ST.wilson(int(d.rule_clause_correct.sum()), len(d)); lo_tr, hi_tr = ST.wilson(int(d.traceability_complete.sum()), len(d))
    fmt = lambda k, lo, hi: f'{k:.3f} [{lo:.3f}, {hi:.3f}]'
    return {'n_cases': f'{len(d)}/{n_planned}', 'response_accuracy': fmt(d.correct.mean(), lo_acc, hi_acc), 'accuracy_complete_evidence': f'{nab.correct.mean():.3f}' if len(nab) else 'n/a',
            'macro_precision': f"{m['macro_precision']:.3f}", 'macro_recall': f"{m['macro_recall']:.3f}", 'weighted_f1': f"{m['weighted_f1']:.3f}", 'macro_f1': f"{m['macro_f1']:.3f}", 'cohen_kappa': f"{m['kappa']:.3f}",
            'rule_correctness': fmt(d.rule_clause_correct.mean(), lo_rule, hi_rule), 'traceability_complete': fmt(d.traceability_complete.mean(), lo_tr, hi_tr),
            'abstention_correct': f"{(ab.predicted_response == 'NE').mean():.3f}" if len(ab) else 'n/a', 'unsupported_decision_rate': f"{ab.unsupported_decision.mean():.3f}" if len(ab) else 'n/a',
            'false_abstention_rate': f"{(nab.predicted_response == 'NE').mean():.3f}" if len(nab) else 'n/a'}


def summarize_real(d):
    m = ST.cls_metrics(d.reference_response, d.predicted_response)
    return {'n_cases': f'{len(d)}/{len(d)}', 'response_accuracy': f'{d.correct.mean():.3f}', 'macro_precision': f"{m['macro_precision']:.3f}", 'macro_recall': f"{m['macro_recall']:.3f}",
            'weighted_f1': f"{m['weighted_f1']:.3f}", 'macro_f1': f"{m['macro_f1']:.3f}", 'cohen_kappa': f"{m['kappa']:.3f}"}


def main():
    raw_all = [json.loads(l) for l in (OUT / 'medvlm_raw.jsonl').read_text().splitlines()]
    cases = pd.read_csv(OUT / 'rule_challenge_cases.csv'); cases_main = cases[cases.subset == 'main']; real = pd.read_csv(OUT / 'real_hcuch_cases.csv')
    n_planned_main = len(cases_main)
    all_pred, table_main, table_real = [], [], []
    for mk, name in MODELS.items():
        raw_main = [j for j in raw_all if j['model_key'] == mk and j.get('source') == 'main' and j.get('run_success')]
        raw_real = [j for j in raw_all if j['model_key'] == mk and j.get('source') == 'real' and j.get('run_success')]
        if raw_main:
            dm = main_subset_rows(mk, name, raw_main, cases_main); all_pred.append(dm)
            table_main.append({'system': name, **summarize_main(dm, n_planned_main)})
        if raw_real:
            dr = real_rows(mk, name, raw_real, real); all_pred.append(dr)
            table_real.append({'system': name, **summarize_real(dr)})
    if all_pred: pd.concat(all_pred, ignore_index=True).to_csv(OUT / 'medvlm_predictions_long.csv', index=False)
    pd.DataFrame(table_main).to_csv(OUT / 'medvlm_main_table.csv', index=False)
    pd.DataFrame(table_real).to_csv(OUT / 'medvlm_real_table.csv', index=False)
    pd.set_option('display.width', 220)
    print('=== synthetic main subset (n planned = 2000) ==='); print(pd.DataFrame(table_main).to_string(index=False))
    print('\n=== real HCUCH cohort (target-lesion response, n = 23) ==='); print(pd.DataFrame(table_real).to_string(index=False))
    return table_main, table_real


if __name__ == '__main__':
    main()
