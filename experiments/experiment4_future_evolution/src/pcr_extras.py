#!/usr/bin/env python3
"""4B extensions on the archived longitudinal-forecasting out-of-fold predictions (no refit of the visit-specific models) + a shared predictor.
Writes results/4B_*.csv.  python pcr_extras.py [--jobs 20]"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

import forecast as FC
from check import check
from common import NB, RES, REPO, SEED, T2, F
from revision import pcr_model

warnings.filterwarnings('ignore', category=ConvergenceWarning)
RUN29 = REPO / 'outputs/longitudinal_forecasting/analysis'


def boot(n):
    rng = np.random.default_rng(SEED)
    return [i for i in (rng.integers(0, n, n) for _ in range(NB))]


def shared_fold(r, te, pts, S, C, ypcr):
    te = set(te); tr = [p for p in pts if p not in te]
    def rows(ps):
        X = []
        for p in ps:
            for k in range(4):
                x = F(S[p][k], k, C[p]); x.update({f'landmark_{j}': float(j == k) for j in range(4)}); x['_p'] = p; x['_k'] = k; X.append(x)
        return pd.DataFrame(X)
    Xtr, Xte = rows(tr), rows(sorted(te))
    cols = [c for c in Xtr.columns if not c.startswith('_')]
    Xte = Xte.reindex(columns=cols + ['_p', '_k'])
    m = pcr_model().fit(Xtr[cols].to_numpy(float), np.array([ypcr[p] for p in Xtr._p]))
    pr = m.predict_proba(Xte[cols].to_numpy(float))[:, 1]
    return [dict(repeat=r, patient_id=p, landmark=T2[k], p=v) for p, k, v in zip(Xte._p, Xte._k, pr)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=20); a = ap.parse_args()
    check()
    O = pd.read_csv(RUN29 / 'oof_predictions.csv', dtype={'patient_id': str})
    O = O[(O.scenario == 'natural') & (O.representation.isin(['S', 'B']))]
    S_ = O[O.representation == 'S'].pivot(index='patient_id', columns='landmark', values='p')[T2]
    y = O[O.representation == 'S'].groupby('patient_id').pcr.first().loc[S_.index].to_numpy()
    out = {}
    # calibration curves (deciles of predicted risk)
    cal = []
    for t in T2:
        p = S_[t].to_numpy(); q = pd.qcut(p, 10, labels=False, duplicates='drop')
        for b in np.unique(q):
            cal.append(dict(landmark=t, bin=int(b), mean_pred=p[q == b].mean(), observed=y[q == b].mean(), n=int((q == b).sum())))
    pd.DataFrame(cal).to_csv(RES / '4B_calibration_bins.csv', index=False)
    # trajectories by final pCR (mean predicted probability with patient-bootstrap CI)
    bi = boot(len(y)); traj = []
    for grp in (0, 1):
        idx = np.where(y == grp)[0]; rg = np.random.default_rng(SEED + grp); bg = [rg.choice(idx, len(idx)) for _ in range(NB)]
        for t in T2:
            v = S_[t].to_numpy()
            traj.append(dict(pcr=grp, landmark=t, mean=v[idx].mean(), lo=np.percentile([v[i].mean() for i in bg], 2.5), hi=np.percentile([v[i].mean() for i in bg], 97.5), n=len(idx)))
    pd.DataFrame(traj).to_csv(RES / '4B_trajectories.csv', index=False)
    d = (S_['T3'] - S_['T0']).to_numpy()
    out['change_T0_T3'] = dict(pcr_mean=float(d[y == 1].mean()), nonpcr_mean=float(d[y == 0].mean()),
                               pcr_increase_frac=float((d[y == 1] > 0).mean()), nonpcr_decrease_frac=float((d[y == 0] < 0).mean()))
    # uncertain-early subgroup: middle tertile of the T0 prediction
    t0 = S_['T0'].to_numpy(); lo_, hi_ = np.quantile(t0, [1 / 3, 2 / 3]); u = (t0 >= lo_) & (t0 <= hi_)
    yu = y[u]; a0, a3 = S_['T0'].to_numpy()[u], S_['T3'].to_numpy()[u]; bu = boot(u.sum())
    dd = [roc_auc_score(yu[i], a3[i]) - roc_auc_score(yu[i], a0[i]) for i in bu if 0 < yu[i].mean() < 1]
    out['uncertain_early'] = dict(n=int(u.sum()), prevalence=float(yu.mean()), auroc_T0=roc_auc_score(yu, a0), auroc_T3=roc_auc_score(yu, a3),
                                  diff=roc_auc_score(yu, a3) - roc_auc_score(yu, a0), lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)))
    # shared predictor (all landmarks stacked, landmark one-hot) vs visit-specific
    I, folds, pts, S, H, C, A, day, yv = FC.load()
    ypcr = {p: int(I.clin.at[p, 'pCR']) for p in pts}
    splits = [(r, list(g.patient_id.astype(str))) for (r, f), g in folds.groupby(['repeat', 'fold'])]
    res = Parallel(n_jobs=a.jobs)(delayed(shared_fold)(r, te, pts, S, C, ypcr) for r, te in splits)
    SH = pd.DataFrame([x for rr in res for x in rr]).groupby(['patient_id', 'landmark']).p.mean().unstack()[T2].loc[S_.index]
    SH.to_csv(RES / '4B_shared_predictor_oof.csv')
    sh = []
    for t in T2:
        a_, b_ = SH[t].to_numpy(), S_[t].to_numpy()
        dd = [roc_auc_score(y[i], a_[i]) - roc_auc_score(y[i], b_[i]) for i in bi]
        sh.append(dict(landmark=t, auroc_shared=roc_auc_score(y, a_), auroc_visit_specific=roc_auc_score(y, b_), diff=roc_auc_score(y, a_) - roc_auc_score(y, b_),
                       lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), n=len(y)))
    pd.DataFrame(sh).to_csv(RES / '4B_shared_vs_specific.csv', index=False)
    (RES / '4B_extras.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)); print(pd.DataFrame(sh).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
