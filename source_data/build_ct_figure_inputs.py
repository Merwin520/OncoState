#!/usr/bin/env python3
"""CT lesion-state figure inputs from the regenerated Experiment 2.2 pipeline.

Reads the S0/A evaluation of the regenerated run (outputs/exp22_baselines/exp22_regen, regen_chain) and its patient-bootstrap statistics
(outputs/exp22_full_final, exp22_full_stats.py) and writes the per-figure inputs used by build_ct_state_data.py:
stats.csv, meta.json, lesions_A.csv, regsanity.csv, match_f1_S0.csv, registration_diagnosis.json (locked cohort, 274 patients).

    python source_data/build_ct_figure_inputs.py
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / 'experiments/exp1_agentic_longitudinal/outputs'
REGEN = EXP / 'exp22_baselines/exp22_regen'
FULL = EXP / 'exp22_full_final'
OUT = Path(__file__).resolve().parent / 'ct_figure_inputs'
LES_COLS = ['gt_class', 'pred_status', 'gt_vol_bl_mL', 'gt_vol_fu_mL', 'pred_vol_bl_mL', 'pred_vol_fu_mL', 'dice_bl', 'dice_fu', 'hd95_mm_bl', 'hd95_mm_fu',
            'gt_diam_bl_mm', 'gt_diam_fu_mm', 'pred_diam_bl_mm', 'pred_diam_fu_mm', 'pred_change_pct', 'gt_change_pct', 'gt_change_class',
            'pred_change_class']


def main():
    OUT.mkdir(exist_ok=True)
    pd.read_csv(FULL / 'exp22_full_statistics.csv').to_csv(OUT / 'stats.csv', index=False)
    (OUT / 'meta.json').write_text((FULL / 'exp22_full_meta.json').read_text())
    la = pd.read_csv(REGEN / 'evalx_lesions_A.csv')
    la[la.cohort == 'locked'][LES_COLS].to_csv(OUT / 'lesions_A.csv', index=False)
    rs = pd.read_csv(REGEN / 'eval_regsanity_S0.csv', dtype={'patient_id': str})
    rs[rs.cohort == 'locked'][['patient_id', 'dist_shared_id_mm', 'dist_nearest_other_fu_lesion_mm']].to_csv(OUT / 'regsanity.csv', index=False)
    L = pd.read_csv(FULL / 'patient_metrics_long_full.csv', dtype={'patient_id': str})
    mf = L[(L.metric == 'match_f1') & (L.setting == 'S0') & (L.cohort == 'locked')][['patient_id', 'value']]
    mf.to_csv(OUT / 'match_f1_S0.csv', index=False)
    # patients with reference persistent lesions but match F1 = 0 under reference masks (S0); F1 is undefined without persistent lesions
    (OUT / 'registration_diagnosis.json').write_text(json.dumps(dict(n_bad=int((mf.value == 0).sum())), indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
