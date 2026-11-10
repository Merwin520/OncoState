#!/usr/bin/env python3
"""Robustness check: within-subtype arm-permutation test for the pCR gain.

Reproduces the pre-specified pCR analysis of run_ispy2.py exactly (same features, same 5 x 5 stratified folds built from the TRUE
arm x pCR strata, same tuning), then refits the state + arm model 200 times with arms permuted among patients WITHIN clinical
subtype (the randomization structure). p = (1 + #{null dAUROC >= observed}) / (1 + 200).
Writes outputs/exp27/ispy2/pcr_permutation.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ispy2 as R  # noqa: E402

FEATS = ['log_ftv', 'log_ld', 'sphericity', 'bpe', 'HR', 'HER2', 'MP', 'age', 'postmeno', 'early_logratio']


def data():
    df = R.load()
    clin = pd.read_excel(R.CLIN).assign(patient_id=lambda d: d.Patient_ID.astype(str)).set_index('patient_id')
    t01 = df[df.k == 0].set_index('patient_id')
    pc = t01[FEATS[:-1] + ['subtype', 'arm']].copy()
    pc['early_logratio'] = t01.y
    return pc.join(clin.pCR, how='inner').dropna(subset=['pCR'])


def splits(pc):
    return [(r, tr, te) for r in range(5)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(pc, pc.arm + pc.pCR.astype(str))]


def oof(pc, arm, folds, arms, with_arm):
    warnings.filterwarnings('ignore')
    preds = np.zeros((5, len(pc)))
    for r, tr, te in folds:
        med = pc.iloc[tr][FEATS].median()

        def mk(idx):
            Z = pc.iloc[idx][FEATS].fillna(med).to_numpy(float)
            return np.c_[Z, np.column_stack([(arm[idx] == a).astype(float) for a in arms])] if with_arm else Z
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=.5, max_iter=20000)),
                          {'logisticregression__C': [.01, .03, .1, .3, 1]}, cv=3, scoring='roc_auc').fit(mk(tr), pc.pCR.iloc[tr])
        preds[r, te] = gs.predict_proba(mk(te))[:, 1]
    return preds.mean(0)


def main():
    pc = data(); y = pc.pCR.to_numpy().astype(int); arm = pc.arm.to_numpy(); arms = sorted(pc.arm.unique())
    folds = splits(pc)
    p0 = oof(pc, arm, folds, arms, False); p1 = oof(pc, arm, folds, arms, True)
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p0)
    print(f'reproduced: AUROC {roc_auc_score(y, p0):.4f} -> {roc_auc_score(y, p1):.4f}, dAUROC {obs:+.4f}', flush=True)

    def one(s):
        rng = np.random.default_rng(s); perm = arm.copy()
        for sub in pc.subtype.unique():
            idx = np.where(pc.subtype.to_numpy() == sub)[0]; perm[idx] = rng.permutation(arm[idx])
        return roc_auc_score(y, oof(pc, perm, folds, arms, True)) - roc_auc_score(y, p0)
    null = np.array(Parallel(n_jobs=100)(delayed(one)(s) for s in range(200)))
    res = dict(label='POST-HOC within-subtype arm permutation (not pre-specified)', n=len(y), d_auroc_observed=float(obs),
               null_mean=float(null.mean()), null_q95=float(np.quantile(null, .95)), null_max=float(null.max()),
               p_one_sided=float((1 + (null >= obs).sum()) / 201), null=null.tolist())
    (R.OUT / 'pcr_permutation.json').write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != 'null'})


if __name__ == '__main__':
    main()
