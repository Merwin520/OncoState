"""State representation ablation: which parts of S_BL carry the future-state signal?

Target: mRECIST % change (primary). No action, no horizon (WM-0 form). Same locked folds, same models, same tuning as
fit_models.py. Paired bootstrap on repeat-averaged OOF predictions. Usage: ablation.py <model run dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import fit_models as FM

MEAS0 = ['bl_mrecist_mm']
MEAS = ['bl_mrecist_mm', 'bl_recist_mm', 'log_bl_easl']
SEG = ['log_seg_mass_ml', 'log1p_seg_mass_components', 'seg_largest_component_frac', 'log_seg_liver_ml', 'seg_mass_to_liver']
CLIN = [f for f in FM.S if f not in MEAS + SEG]
SETS = {'S1 baseline mRECIST only': MEAS0, 'S2 baseline measurements (mRECIST, RECIST, EASL)': MEAS,
        'S3 measurements + SEG burden': MEAS + SEG, 'S4 full S_BL (measurements + SEG + clinical)': MEAS + SEG + CLIN,
        'X1 SEG burden only': SEG, 'X2 clinical only': CLIN, 'X3 measurements + clinical': MEAS + CLIN}
N_BOOT, SEED = 2000, 20260925


def main():
    run = Path(sys.argv[1])
    assert FM.S == MEAS + SEG + CLIN or set(FM.S) == set(MEAS + SEG + CLIN)
    sp = json.loads((FM.SPLITS / 'splits.json').read_text())
    assert sp['audit_passed']
    _, fr = FM.state.build()
    tasks = [dict(id=i, kind='reg', target='y_pct', model=m, variant=v, variant_label=v, cols=SETS[v])
             for i, (m, v) in enumerate((m, v) for m in ('enet', 'gbr') for v in SETS)]
    res = Parallel(n_jobs=64)(delayed(FM.run_task)(fr, f, t) for t in tasks for f in sp['folds'])
    pr = pd.DataFrame([r for rs in res for r in rs])
    out = run / 'ablation'
    out.mkdir(exist_ok=True)
    pr.to_csv(out / 'oof_predictions.csv.gz', index=False)
    avg = pr.groupby(['model', 'variant', 'patient_id']).agg(y=('y_true', 'first'), p=('y_pred', 'mean')).reset_index()
    main_pr = pd.read_csv(run / 'oof_predictions.csv.gz')
    naive = main_pr[(main_pr.model == 'train_mean') & (main_pr.target == 'y_pct')].groupby('patient_id').y_pred.mean().sort_index()
    idx = np.random.default_rng(SEED).integers(0, len(naive), (N_BOOT, len(naive)))
    rows, comps = [], []
    for m in ('enet', 'gbr'):
        V = {v: avg[(avg.model == m) & (avg.variant == v)].set_index('patient_id').sort_index() for v in SETS}
        y = V['S4 full S_BL (measurements + SEG + clinical)'].y.to_numpy()
        e = {v: np.abs(d.p.to_numpy() - y) for v, d in V.items()}
        en = np.abs(naive.to_numpy() - y)
        for v, d in V.items():
            b = e[v][idx].mean(1)
            dn = (e[v][idx] - en[idx]).mean(1)
            rows.append(dict(model=m, state=v, n_features=len(SETS[v]), MAE=e[v].mean(), MAE_lo=np.percentile(b, 2.5), MAE_hi=np.percentile(b, 97.5),
                             spearman=spearmanr(y, d.p).statistic, orr_auroc=roc_auc_score(y <= -30, -d.p),
                             vs_naive=(e[v] - en).mean(), vs_naive_lo=np.percentile(dn, 2.5), vs_naive_hi=np.percentile(dn, 97.5)))
        for a, b_ in (('S4 full S_BL (measurements + SEG + clinical)', 'S1 baseline mRECIST only'),
                      ('S4 full S_BL (measurements + SEG + clinical)', 'S2 baseline measurements (mRECIST, RECIST, EASL)'),
                      ('S3 measurements + SEG burden', 'S2 baseline measurements (mRECIST, RECIST, EASL)'),
                      ('S4 full S_BL (measurements + SEG + clinical)', 'S3 measurements + SEG burden'),
                      ('S4 full S_BL (measurements + SEG + clinical)', 'X3 measurements + clinical'),
                      ('S4 full S_BL (measurements + SEG + clinical)', 'X2 clinical only')):
            d = (e[a][idx] - e[b_][idx]).mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            comps.append(dict(model=m, comparison=f'{a}  vs  {b_}', dMAE=(e[a] - e[b_]).mean(), ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
    pd.DataFrame(rows).to_csv(out / 'state_ablation.csv', index=False)
    pd.DataFrame(comps).to_csv(out / 'state_ablation_paired.csv', index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(comps).round(3).to_string(index=False))
    # the full-state set must reproduce WM-0 from the main run
    w0 = main_pr[(main_pr.variant_label == 'WM-0') & (main_pr.target == 'y_pct') & main_pr.model.isin(['enet', 'gbr'])]
    full = pr[pr.variant == 'S4 full S_BL (measurements + SEG + clinical)']
    j = full.merge(w0, on=['model', 'repeat', 'patient_id'])
    print('S4 vs main-run WM-0 max |diff|:', float((j.y_pred_x - j.y_pred_y).abs().max()), len(j))


if __name__ == '__main__':
    main()
