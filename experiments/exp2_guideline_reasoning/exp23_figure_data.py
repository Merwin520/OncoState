"""Experiment 2.3 source data (no plotting): the combined figure3_data.csv tagged by a 'panel' column.
A real HCUCH target-lesion reconstruction; B synthetic benchmark matrix; C new-lesion override case; D threshold stress curves.
Same computations as the original figure script, without the drawing code.
    python exp23_figure_data.py   -> outputs/exp23_final/figure3_data.csv"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
SYS = ['Full OncoAgent', 'No-retrieval', 'LLM-only', 'Rule executor only']


def main():
    stats = pd.read_csv(OUT / 'exp23_statistics.csv'); real = pd.read_csv(OUT / 'real_hcuch_cases.csv'); cases = pd.read_csv(OUT / 'rule_challenge_cases.csv')
    clauses = json.loads((OUT / 'guideline_retrieval.json').read_text())['clauses']
    parts = []

    # A: real HCUCH target-lesion response reconstruction
    r = real.sort_values(['patient_id', 'followup_study_date']).reset_index(drop=True)
    mask_chg = (r.mask_derived_followup_sum - r.mask_derived_baseline_sum) / r.mask_derived_baseline_sum * 100
    parts.append(r.assign(mask_change_from_baseline=mask_chg, panel='A')[['panel', 'patient_id', 'study_pair', 'change_from_baseline', 'change_from_nadir',
                                                                         'reference_response', 'predicted_response', 'mask_derived_response',
                                                                         'mask_change_from_baseline']])

    # B: synthetic benchmark matrix (estimates per system and metric)
    rows = [('accuracy', 'Response accuracy'), ('acc_complete', 'Accuracy, complete evidence'), ('abstention_correct', 'Abstains when evidence missing'),
            ('trace', 'Traceability complete'), ('cit_supported', 'Citation supported')]

    def est(k, s):
        q = stats[(stats.section == 'synthetic_main') & (stats.metric == k) & (stats.system == s)]
        return q.estimate.iloc[0] if not q.empty else np.nan
    M = np.array([[est(k, s) for s in SYS] for k, _ in rows], float)
    parts.append(pd.DataFrame([dict(row=lab, metric=k, system=s, value=M[i, j]) for i, (k, lab) in enumerate(rows) for j, s in enumerate(SYS)]).assign(panel='B'))

    # C: rule-challenge case (new lesion overrides target response)
    cand = cases[(cases.challenge_type == 'D_new_lesion_override') & (cases.n_target_lesions >= 2)].iloc[0]
    hist = json.loads(cand.history_json); base_s, cur = hist[0], cand.current_sum; chg = (cur - base_s) / base_s * 100
    cl = clauses['R.O.NEW']; cite = cl['citations'][0]['title'] if cl['citations'] else 'RECIST 1.1'
    parts.append(pd.DataFrame([dict(panel='C', case_id=cand.case_id, template=cand.template_source, baseline_sum=base_s, current_sum=cur, change_pct=chg,
                                    new_lesion=cand.new_lesion, expected=cand.expected_response, rule_ids=cand.expected_rule_ids, clause_source=cite,
                                    clause_text=cl['text'][:400])]))

    # D: threshold stress, proportion of decisions in the threshold-crossed category (Wilson 95% CI)
    st = pd.read_csv(OUT / 'threshold_stress_results.csv')
    FAM = [('S1_PR', 'PR'), ('S2_PD', 'PD'), ('S3_ABS', 'PD')]
    rows_d = []
    for fam, cls in FAM:
        d = st[st.stress_family == fam].copy(); d['offset'] = (d.series_pos - 2) * 0.1
        offs = np.array(sorted(d.offset.unique()))
        for sysn in ('Full OncoAgent', 'LLM-only'):
            for o in offs:
                x = d[(d.system == sysn) & (np.isclose(d.offset, o))]; k, n = int((x.predicted == cls).sum()), len(x)
                p_ = k / n; z = 1.96; c = (p_ + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                rows_d.append(dict(family=fam, offset_mm=round(o, 2), system=sysn, proportion_in_crossed_category=p_, n=n, ci95_lo=max(0, c - h),
                                   ci95_hi=min(1, c + h)))
    parts.append(pd.DataFrame(rows_d).assign(panel='D'))

    cols = ['panel'] + sorted({c for d in parts for c in d.columns if c != 'panel'})
    pd.concat(parts, ignore_index=True, sort=False)[cols].to_csv(OUT / 'figure3_data.csv', index=False)
    print('wrote', OUT / 'figure3_data.csv')


if __name__ == '__main__':
    main()
