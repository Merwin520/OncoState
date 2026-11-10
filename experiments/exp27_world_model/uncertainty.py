"""Uncertainty and calibration of the one-step future-state prediction.

For every outer fold (locked splits): tune as in fit_models.py, then obtain inner-fold out-of-fold residuals on the
TRAINING patients only (locked inner folds, best hyperparameters). These residuals give
  * split-conformal-style prediction intervals for the test patients: pred +/- q_level(|r|), finite-sample quantile,
    lower bound truncated at -100 % (the physical minimum of % change);
  * a model-derived response probability P(ORR) = P(change <= -30 %) = mean_i[pred + r_i <= -30] over signed residuals.
Test patients never contribute to their own interval or probability. Output: <model run>/uncertainty/.
Usage: uncertainty.py <model run dir>
"""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV

import fit_models as FM

LEVELS = (0.5, 0.9)
THRESH = -30.0


def conformal_q(abs_r, level):
    n = len(abs_r)
    k = min(n, math.ceil((n + 1) * level))
    return float(np.sort(abs_r)[k - 1])


def run_fold(fr, fold, model, variant):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    cols = FM.VARIANTS[variant]
    tr, te = fold['train'], fold['test']
    X_tr, y_tr, X_te = fr.loc[tr, cols], fr.loc[tr, 'y_pct'].to_numpy(), fr.loc[te, cols]
    pos = {p: i for i, p in enumerate(tr)}
    inner = [(np.array([pos[p] for p in f['train']]), np.array([pos[p] for p in f['valid']])) for f in fold['inner']]
    est, grid = FM.regressor(model)
    gs = GridSearchCV(est, grid, cv=inner, scoring='neg_mean_absolute_error', n_jobs=1).fit(X_tr, y_tr)
    r = np.empty(len(tr))
    for a, b in inner:  # inner out-of-fold residuals with the selected hyperparameters (training patients only)
        e = clone(gs.best_estimator_).fit(X_tr.iloc[a], y_tr[a])
        r[b] = y_tr[b] - e.predict(X_tr.iloc[b])
    pred = gs.predict(X_te)
    out = []
    for p, v in zip(te, pred):
        row = dict(model=model, variant=variant, repeat=fold['repeat'], fold=fold['fold'], patient_id=p, y_true=fr.at[p, 'y_pct'],
                   y_pred=float(v), p_orr=float(np.mean(v + r <= THRESH)), n_calib=len(r))
        for lv in LEVELS:
            q = conformal_q(np.abs(r), lv)
            # change cannot fall below -100 % (complete disappearance); truncation cannot reduce coverage since y >= -100.
            # A point prediction itself may lie slightly below -100 (3/1640, left unclipped as in the primary analysis);
            # the lower bound then equals the point prediction so that lo <= pred always holds.
            row[f'lo{int(lv * 100)}'], row[f'hi{int(lv * 100)}'] = min(max(v - q, -100.0), v), v + q
        out.append(row)
    return out


def ece(y, p, bins=5):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def cal_slope(y, p):
    lp = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def main():
    run = Path(sys.argv[1])
    sp = json.loads((FM.SPLITS / 'splits.json').read_text())
    assert sp['audit_passed']
    _, fr = FM.state.build()
    jobs = [(m, v) for m in ('enet', 'gbr') for v in ('WM-0', 'WM-3')]
    res = Parallel(n_jobs=64)(delayed(run_fold)(fr, f, m, v) for m, v in jobs for f in sp['folds'])
    df = pd.DataFrame([r for rs in res for r in rs])
    out = run / 'uncertainty'
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'intervals_per_repeat.csv.gz', index=False)

    rows, thr, calib = [], [], []
    for (m, v), g in df.groupby(['model', 'variant']):
        # consistency with the main run (same tuning, same folds => same point predictions)
        per_rep = g.groupby('repeat')
        cov = {lv: per_rep.apply(lambda x: ((x.y_true >= x[f'lo{lv}']) & (x.y_true <= x[f'hi{lv}'])).mean(), include_groups=False) for lv in (50, 90)}
        wid = {lv: (g[f'hi{lv}'] - g[f'lo{lv}']).mean() for lv in (50, 90)}
        a = g.groupby('patient_id').agg(y=('y_true', 'first'), p=('y_pred', 'mean'), porr=('p_orr', 'mean'),
                                        lo90=('lo90', 'mean'), hi90=('hi90', 'mean')).sort_index()
        yb = (a.y <= THRESH).astype(int).to_numpy()
        ci_, cs_ = cal_slope(yb, a.porr.to_numpy())
        lin = np.polyfit(a.p, a.y, 1)
        rows.append(dict(model=m, variant=v, coverage50_mean=cov[50].mean(), coverage50_sd=cov[50].std(), width50=wid[50],
                         coverage90_mean=cov[90].mean(), coverage90_sd=cov[90].std(), width90=wid[90],
                         continuous_calibration_slope=lin[0], continuous_calibration_intercept=lin[1],
                         orr_brier=brier_score_loss(yb, a.porr), orr_brier_prevalence=brier_score_loss(yb, np.full(len(yb), yb.mean())),
                         orr_ece_5bins=ece(yb, a.porr.to_numpy()), orr_cal_intercept=ci_, orr_cal_slope=cs_,
                         orr_auroc_p=roc_auc_score(yb, a.porr), orr_auroc_pred_change=roc_auc_score(yb, -a.p)))
        # threshold distance: how far is the predicted change from the -30 % response threshold?
        dist = (a.p - THRESH).abs()
        pred_orr = (a.p <= THRESH).astype(int).to_numpy()
        for lo, hi in ((0, 10), (10, 25), (25, np.inf)):
            s = ((dist >= lo) & (dist < hi)).to_numpy()
            thr.append(dict(model=m, variant=v, distance_bin=f'[{lo}, {hi})', n=int(s.sum()),
                            rule_accuracy=float((pred_orr[s] == yb[s]).mean()) if s.any() else np.nan,
                            observed_orr_rate=float(yb[s].mean()) if s.any() else np.nan,
                            mean_p_orr=float(a.porr[s].mean()) if s.any() else np.nan))
        straddle = ((a.lo90 <= THRESH) & (a.hi90 >= THRESH)).to_numpy()
        conf = ~straddle
        sens = lambda msk: float(((pred_orr == 1) & (yb == 1) & msk).sum() / max(((yb == 1) & msk).sum(), 1))
        spec = lambda msk: float(((pred_orr == 0) & (yb == 0) & msk).sum() / max(((yb == 0) & msk).sum(), 1))
        allm = np.ones(len(yb), bool)
        # probability-based abstention: uncertain if 0.2 < P(ORR) < 0.8
        pconf = ~((a.porr > .2) & (a.porr < .8)).to_numpy()
        pl = (a.porr >= .5).astype(int).to_numpy()
        calib.append(dict(model=m, variant=v, n=len(yb), all_sens=sens(allm), all_spec=spec(allm), all_acc=float((pred_orr == yb).mean()),
                          pi90_uncertain_n=int(straddle.sum()), pi90_confident_n=int(conf.sum()),
                          pi90_confident_acc=float((pred_orr[conf] == yb[conf]).mean()) if conf.any() else np.nan,
                          pi90_confident_sens=sens(conf), pi90_confident_spec=spec(conf),
                          pi90_confident_nonORR=int(((yb == 0) & conf).sum()),
                          p_uncertain_n=int((~pconf).sum()), p_confident_acc=float((pl[pconf] == yb[pconf]).mean()) if pconf.any() else np.nan,
                          p_confident_sens=float(((pl == 1) & (yb == 1) & pconf).sum() / max(((yb == 1) & pconf).sum(), 1)),
                          p_confident_spec=float(((pl == 0) & (yb == 0) & pconf).sum() / max(((yb == 0) & pconf).sum(), 1)),
                          p_confident_nonORR=int(((yb == 0) & pconf).sum())))
        # calibration curve data (5 equal-frequency bins)
        q = pd.qcut(a.porr.rank(method='first'), 5, labels=False)
        pd.DataFrame({'bin': range(5), 'mean_p_orr': a.porr.groupby(q).mean().values, 'observed_orr': pd.Series(yb, index=a.index).groupby(q).mean().values,
                      'n': q.value_counts().sort_index().values}).to_csv(out / f'calibration_curve_{m}_{v}.csv', index=False)
    for name, d in (('interval_calibration', rows), ('threshold_distance', thr), ('abstention', calib)):
        pd.DataFrame(d).to_csv(out / f'{name}.csv', index=False)
        print(pd.DataFrame(d).round(3).to_string(index=False))
    # point predictions must reproduce the main run
    main_pr = pd.read_csv(run / 'oof_predictions.csv.gz')
    mp = main_pr[(main_pr.target == 'y_pct') & (main_pr.shuffle_seed == -1) & main_pr.model.isin(['enet', 'gbr']) & main_pr.variant_label.isin(['WM-0', 'WM-3'])]
    j = df.merge(mp, left_on=['model', 'variant', 'repeat', 'patient_id'], right_on=['model', 'variant_label', 'repeat', 'patient_id'])
    print('max |pred - main-run pred|:', float((j.y_pred_x - j.y_pred_y).abs().max()), 'rows', len(j))


if __name__ == '__main__':
    main()
