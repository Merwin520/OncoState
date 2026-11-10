#!/usr/bin/env python3
"""Source data: deterministic RECIST 1.1 rule fidelity.

    python build_recist_rule_data.py
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / 'experiments/exp2_guideline_reasoning/outputs/exp23_final'
SYSTEMS = ['LLaVA-Med v1.5 7B', 'MedGemma 4B-it', 'Full OncoAgent']   # systems with case-level predictions
RADAR = [('response_accuracy', 'Accuracy'), ('macro_precision', 'Macro\nprecision'), ('macro_recall', 'Macro\nrecall'),
         ('weighted_f1', 'Weighted F1'), ('macro_f1', 'Macro F1'), ('cohen_kappa', "Cohen's κ"), ('rule_correctness', 'Reference-rule\ncorrectness' if __import__('os').environ.get('FIG_FINAL') else 'Rule\ncorrectness')]


def _ci(cell):
    m = re.match(r'\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]', str(cell))
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (float(cell), np.nan, np.nan)


def build() -> pd.DataFrame:
    R = []

    def r(panel, series='', label='', x=np.nan, y=np.nan, lo=np.nan, hi=np.nan, value=np.nan, n=np.nan, note=''):
        R.append(dict(panel=panel, series=series, label=label, x=x, y=y, lo=lo, hi=hi, value=value, n=n, note=note))

    main = pd.read_csv(EXP / 'exp23_main_table.csv').set_index('system')
    vlm = pd.read_csv(EXP / 'medvlm_main_table.csv')
    vlm['system'] = vlm.system.str.extract(r'\((.*?), text-only\)')[0]
    vlm = vlm.set_index('system')
    src = {'Full OncoAgent': (main, 'experiments/exp2_guideline_reasoning/outputs/exp23_final/exp23_main_table.csv')}
    for s in ('LLaVA-Med v1.5 7B', 'MedGemma 4B-it'):
        src[s] = (vlm, 'experiments/exp2_guideline_reasoning/outputs/exp23_final/medvlm_main_table.csv')
    for s in SYSTEMS:
        t, note = src[s]
        for col, _ in RADAR:
            v, lo, hi = _ci(t.at[s, col])
            if col != 'response_accuracy':
                lo = hi = np.nan
            r('bcd', s, col, value=v, lo=lo, hi=hi, n=2000, note=note)
    f3 = pd.read_csv(EXP / 'figure3_data.csv')
    for row in f3[f3.panel == 'A'].itertuples():
        pair = row.study_pair.split('-')[-1]
        r('e', row.predicted_response, f"P{int(row.patient_id)}\nF{pair}", y=row.change_from_baseline, value=row.mask_change_from_baseline,
          note=f'reference {row.reference_response}; mask-derived {row.mask_derived_response}')
    order = ['Response accuracy', 'Accuracy, complete evidence', 'Abstains when evidence missing', 'Traceability complete', 'Citation supported']
    for row in f3[f3.panel == 'B'].itertuples():
        r('f', row.system, row.row, x=['Full OncoAgent', 'No-retrieval', 'LLM-only', 'Rule executor only'].index(row.system),
          y=order.index(row.row), value=row.value)
    c = f3[f3.panel == 'C'].iloc[0]
    case = pd.read_csv(EXP / 'rule_challenge_cases.csv').set_index('case_id').loc[c.case_id]
    for k, v in dict(case_id=c.case_id, template=case.template_source, baseline_sum=case.baseline_sum, nadir_sum=case.nadir_sum,
                     current_sum=case.current_sum, change_baseline_pct=case.change_baseline_pct, n_target=case.n_target_lesions,
                     new_lesion='present' if case.new_lesion else 'absent', non_target=str(case.non_target_status).lower(),
                     expected=case.expected_response, rule=case.expected_rule_ids, clause=c.clause_text, source=c.clause_source).items():
        r('g', 'case', k, note=str(v))
    for row in f3[f3.panel == 'D'].itertuples():
        r('h', row.system, row.family, x=row.offset_mm, y=row.proportion_in_crossed_category, lo=row.ci95_lo, hi=row.ci95_hi, n=row.n)
    return pd.DataFrame(R)


def main():
    out = Path(__file__).resolve().parent / 'out' / 'figure3_nature_data.csv'
    out.parent.mkdir(exist_ok=True)
    d = build()
    d.to_csv(out, index=False)
    print(f'wrote {out} ({len(d)} rows)')


if __name__ == '__main__':
    main()
