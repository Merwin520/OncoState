#!/usr/bin/env python3
"""4E HCC-TACE-Seg: re-evaluate the ARCHIVED Exp 2.7 out-of-fold predictions (no refit) with the unchanged evaluator definitions
(per-patient mean over the 5 CV repeats; 2,000 patient bootstrap, seed 20260924). Adds paired differences vs the training-mean baseline
and the state-only (WM-0) vs state+treatment+horizon (WM-3) contrast.  python tace.py"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import REPO, RES

RUN = REPO / 'outputs/exp27/models/run'
SEED, NB = 20260924, 2000


def main():
    pr = pd.read_csv(RUN / 'oof_predictions.csv.gz')
    pr = pr[(pr.target == 'y_pct') & (pr.shuffle_seed == -1)]
    avg = pr.groupby(['kind', 'model', 'variant_label', 'patient_id']).agg(y=('y_true', 'first'), p=('y_pred', 'mean'), arm=('arm', 'first')).reset_index()
    pts = sorted(avg.patient_id.unique()); rng = np.random.default_rng(SEED); bi = rng.integers(0, len(pts), size=(NB, len(pts)))
    V = {(k, m, v): g.set_index('patient_id').loc[pts] for (k, m, v), g in avg.groupby(['kind', 'model', 'variant_label'])}
    rows = []
    for (k, m, v), g in V.items():
        y, p = g.y.to_numpy(), g.p.to_numpy(); e = np.abs(p - y)
        mae_b = e[bi].mean(1)
        sp = spearmanr(y, p).statistic if np.std(p) > 0 else np.nan
        orr = (y <= -30).astype(int)
        rows.append(dict(system=f'{m} {v}' if k != 'naive' else f'naive: {m}', n=len(y), n_ctace=int((g.arm == 0).sum()), n_deb=int((g.arm == 1).sum()),
                         MAE=e.mean(), MAE_lo=np.percentile(mae_b, 2.5), MAE_hi=np.percentile(mae_b, 97.5), spearman=sp,
                         ORR_AUROC_rule=roc_auc_score(orr, -p) if np.std(p) > 0 else np.nan))
    T = pd.DataFrame(rows).sort_values('MAE'); T.to_csv(RES / '4E_tace_reproduction.csv', index=False)
    base = V[('naive', 'train_mean', 'naive')]
    pair = []
    for lab, a_, b_ in (('GBM WM-3 - training mean', V[('reg', 'gbr', 'WM-3')], base), ('GBM WM-0 (state only) - training mean', V[('reg', 'gbr', 'WM-0')], base),
                        ('GBM WM-3 - WM-0 (treatment + horizon)', V[('reg', 'gbr', 'WM-3')], V[('reg', 'gbr', 'WM-0')]),
                        ('GBM WM-3 - Elastic Net WM-3', V[('reg', 'gbr', 'WM-3')], V[('reg', 'enet', 'WM-3')])):
        ea, eb = np.abs(a_.p - a_.y).to_numpy(), np.abs(b_.p - b_.y).to_numpy(); d = (ea - eb)[bi].mean(1)
        pair.append(dict(comparison=lab, dMAE=(ea - eb).mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), n=len(ea)))
    P = pd.DataFrame(pair); P.to_csv(RES / '4E_tace_paired.csv', index=False)
    pd.set_option('display.width', 220); print(T.round(3).to_string(index=False)); print(P.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
