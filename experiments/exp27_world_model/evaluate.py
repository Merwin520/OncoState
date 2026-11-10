"""Experiment 2.7: evaluate out-of-fold predictions from fit_models.py.

Per patient, predictions are averaged over the 5 CV repeats (each repeat gives one OOF prediction per patient); per-repeat
metrics are also reported (mean, sd). CIs: 2000 patient-level bootstrap resamples; model comparisons are paired on the
same resamples. Response is derived by the rule layer from predicted % change (ORR <=> change <= -30 %, as verified in
the audit); direct ORR classifiers are secondary. Usage: evaluate.py <model run dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)

N_BOOT, SEED = 2000, 20260924
KEY = ['target', 'kind', 'model', 'variant_label', 'shuffle_seed']


def direction(pct):
    return np.where(pct <= -20, 'decrease', np.where(pct >= 20, 'increase', 'stable'))


def cont_metrics(y, p, bl=None):
    e = p - y
    d = dict(MAE=np.abs(e).mean(), MedAE=np.median(np.abs(e)), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(),
             pearson=pearsonr(y, p).statistic if np.std(p) > 0 else np.nan,
             spearman=spearmanr(y, p).statistic if np.std(p) > 0 else np.nan,
             direction_agreement=(direction(y) == direction(p)).mean(),
             direction_kappa=cohen_kappa_score(direction(y), direction(p)) if len(set(direction(p))) > 1 else 0.0)
    if bl is not None:  # secondary: follow-up mRECIST measurement implied by the predicted change
        fu, fu_hat = bl * (1 + y / 100), np.clip(bl * (1 + p / 100), 0, None)
        d.update(FU_MAE_mm=np.abs(fu_hat - fu).mean(), FU_spearman=spearmanr(fu, fu_hat).statistic)
    return d


def bin_metrics(y, score, yhat, prob=None):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    d = dict(AUROC=roc_auc_score(y, score) if np.std(score) > 0 else 0.5, AUPRC=average_precision_score(y, score),
             balanced_accuracy=balanced_accuracy_score(y, yhat), macro_F1=f1_score(y, yhat, average='macro'),
             sensitivity=tp / (tp + fn), specificity=tn / (tn + fp) if tn + fp else np.nan, kappa=cohen_kappa_score(y, yhat),
             TP=tp, FN=fn, FP=fp, TN=tn, Brier=brier_score_loss(y, prob) if prob is not None else np.nan)
    return d


def main():
    run = Path(sys.argv[1])
    pr = pd.read_csv(run / 'oof_predictions.csv.gz')
    fr = pd.read_csv(run / 'analysis_frame.csv', index_col='patient_id')
    avg = pr.groupby(KEY + ['patient_id'], dropna=False).agg(y_true=('y_true', 'first'), y_pred=('y_pred', 'mean')).reset_index()
    rng = np.random.default_rng(SEED)
    boots = {}

    def vec(target, kind, model, variant, seed=-1):
        a = avg[(avg.target == target) & (avg.kind == kind) & (avg.model == model) & (avg.variant_label == variant) & (avg.shuffle_seed == seed)]
        return a.set_index('patient_id').sort_index()

    def bidx(n):
        if n not in boots:
            boots[n] = rng.integers(0, n, size=(N_BOOT, n))
        return boots[n]

    def ci(fn, *arrs):
        n = len(arrs[0])
        vals = np.array([fn(*[a[i] for a in arrs]) for i in bidx(n)])
        return np.nanpercentile(vals, [2.5, 97.5])

    out = {}
    # ---------- primary continuous table (mRECIST % change) + rule-layer ORR
    rows, brows = [], []
    systems = [('naive', m, 'naive') for m in ('train_mean', 'train_median', 'no_change', 'train_arm_mean')] + \
              [('reg', m, v) for m in ('enet', 'gbr', 'mlp') for v in ('WM-0', 'WM-1', 'WM-2', 'WM-3')]
    for kind, m, v in systems:
        a = vec('y_pct', kind, m, v)
        y, p, bl = a.y_true.to_numpy(), a.y_pred.to_numpy(), fr.loc[a.index, 'bl_mrecist_mm'].to_numpy()
        cm = cont_metrics(y, p, bl)
        lo, hi = ci(lambda yy, pp: np.abs(pp - yy).mean(), y, p)
        slo, shi = ci(lambda yy, pp: spearmanr(yy, pp).statistic if np.std(pp) > 0 else np.nan, y, p)
        per_rep = pr[(pr.target == 'y_pct') & (pr.kind == kind) & (pr.model == m) & (pr.variant_label == v)].groupby('repeat').apply(
            lambda g: np.abs(g.y_pred - g.y_true).mean(), include_groups=False)
        rows.append(dict(system=f'{m} {v}' if kind != 'naive' else f'naive: {m}', model=m, variant=v, N=len(a), **cm, MAE_lo=lo, MAE_hi=hi,
                         spearman_lo=slo, spearman_hi=shi, MAE_per_repeat_mean=per_rep.mean(), MAE_per_repeat_sd=per_rep.std()))
        yb = (y <= -30).astype(int)
        bm = bin_metrics(yb, -p, (p <= -30).astype(int))
        alo, ahi = ci(lambda yy, pp: roc_auc_score(yy, -pp) if len(set(yy)) > 1 and np.std(pp) > 0 else np.nan, yb, p)
        brows.append(dict(system=rows[-1]['system'], source='rule layer on predicted % change', N=len(a), **bm, AUROC_lo=alo, AUROC_hi=ahi))
    cont = pd.DataFrame(rows)
    # ---------- direct ORR classifiers (secondary)
    for m in ('logit', 'gbc'):
        for v in ('WM-0', 'WM-1', 'WM-2', 'WM-3'):
            a = vec('orr', 'clf', m, v)
            y, p = a.y_true.to_numpy().astype(int), a.y_pred.to_numpy()
            bm = bin_metrics(y, p, (p >= .5).astype(int), p)
            alo, ahi = ci(lambda yy, pp: roc_auc_score(yy, pp) if len(set(yy)) > 1 else np.nan, y, p)
            brows.append(dict(system=f'{m} {v}', source='direct classifier (threshold 0.5)', N=len(a), **bm, AUROC_lo=alo, AUROC_hi=ahi))
    binr = pd.DataFrame(brows)

    # ---------- paired comparisons (bootstrap on the same resamples)
    comps = []

    def paired(label, target, ka, ma, va, kb, mb, vb, metric='MAE'):
        a, b = vec(target, ka, ma, va), vec(target, kb, mb, vb)
        assert (a.index == b.index).all()
        y, pa, pb = a.y_true.to_numpy(), a.y_pred.to_numpy(), b.y_pred.to_numpy()
        if metric == 'MAE':
            f = lambda yy, x1, x2: np.abs(x1 - yy).mean() - np.abs(x2 - yy).mean()
        else:  # rule-layer AUROC difference
            f = lambda yy, x1, x2: (roc_auc_score(yy <= -30, -x1) - roc_auc_score(yy <= -30, -x2)) if 0 < (yy <= -30).mean() < 1 else np.nan
        est, (lo, hi) = f(y, pa, pb), ci(f, y, pa, pb)
        comps.append(dict(comparison=label, target=target, metric=f'{metric} difference (A - B)', N=len(a), estimate=est, ci_lo=lo, ci_hi=hi,
                          ci_excludes_0=bool(lo > 0 or hi < 0)))

    for m in ('enet', 'gbr', 'mlp'):
        paired(f'{m}: WM-3 vs WM-0 (treatment + time conditioning)', 'y_pct', 'reg', m, 'WM-3', 'reg', m, 'WM-0')
        paired(f'{m}: WM-1 vs WM-0 (treatment)', 'y_pct', 'reg', m, 'WM-1', 'reg', m, 'WM-0')
        paired(f'{m}: WM-2 vs WM-0 (time)', 'y_pct', 'reg', m, 'WM-2', 'reg', m, 'WM-0')
        paired(f'{m}: WM-3 vs WM-2 (treatment, given time)', 'y_pct', 'reg', m, 'WM-3', 'reg', m, 'WM-2')
        paired(f'{m}: WM-0 vs naive training mean', 'y_pct', 'reg', m, 'WM-0', 'naive', 'train_mean', 'naive')
        paired(f'{m}: WM-3 vs naive training mean', 'y_pct', 'reg', m, 'WM-3', 'naive', 'train_mean', 'naive')
        paired(f'{m}: WM-3 vs WM-0, rule-layer ORR', 'y_pct', 'reg', m, 'WM-3', 'reg', m, 'WM-0', metric='AUROC')
    paired('naive: training treatment-arm mean vs training mean', 'y_pct', 'naive', 'train_arm_mean', 'naive', 'naive', 'train_mean', 'naive')

    # ---------- treatment-shuffle negative control
    shuf = []
    for m in ('enet', 'gbr'):
        real = vec('y_pct', 'reg', m, 'WM-3')
        y, p3 = real.y_true.to_numpy(), real.y_pred.to_numpy()
        P = np.stack([vec('y_pct', 'reg', m, 'WM-shuffle', s).y_pred.to_numpy() for s in range(20)])
        mae_s = np.abs(P - y).mean(1)
        f = lambda yy, x, PP: np.abs(x - yy).mean() - np.abs(PP - yy).mean(1).mean()
        vals = np.array([f(y[i], p3[i], P[:, i]) for i in bidx(len(y))])
        mae3 = np.abs(p3 - y).mean()
        shuf.append(dict(model=m, WM3_MAE=mae3, shuffle_MAE_mean=mae_s.mean(), shuffle_MAE_min=mae_s.min(), shuffle_MAE_max=mae_s.max(),
                         diff_WM3_minus_shuffle=mae3 - mae_s.mean(), ci_lo=np.percentile(vals, 2.5), ci_hi=np.percentile(vals, 97.5),
                         n_shuffle=len(mae_s), n_shuffle_as_good_or_better=int((mae_s <= mae3).sum()),
                         permutation_p=(1 + int((mae_s <= mae3).sum())) / (1 + len(mae_s))))
    shuf = pd.DataFrame(shuf)

    # ---------- reader sensitivity
    rs = []
    for rd in (1, 2, 3):
        t = f'y_pct_r{rd}'
        nm = vec(t, 'naive', 'train_mean', 'naive')
        for m in ('enet', 'gbr'):
            a0, a3 = vec(t, 'reg', m, 'WM-0'), vec(t, 'reg', m, 'WM-3')
            y = a0.y_true.to_numpy()
            f = lambda yy, x1, x2: np.abs(x1 - yy).mean() - np.abs(x2 - yy).mean()
            lo, hi = ci(f, y, a3.y_pred.to_numpy(), a0.y_pred.to_numpy())
            nlo, nhi = ci(f, y, a0.y_pred.to_numpy(), nm.y_pred.to_numpy())
            rs.append(dict(reader=rd, model=m, N=len(y), naive_MAE=np.abs(nm.y_pred - nm.y_true).mean(),
                           WM0_MAE=np.abs(a0.y_pred - a0.y_true).mean(), WM3_MAE=np.abs(a3.y_pred - a3.y_true).mean(),
                           WM3_minus_WM0=f(y, a3.y_pred.to_numpy(), a0.y_pred.to_numpy()), ci_lo=lo, ci_hi=hi,
                           WM0_minus_naive=f(y, a0.y_pred.to_numpy(), nm.y_pred.to_numpy()), naive_ci_lo=nlo, naive_ci_hi=nhi,
                           WM0_spearman=spearmanr(a0.y_true, a0.y_pred).statistic, WM3_spearman=spearmanr(a3.y_true, a3.y_pred).statistic))
    rs = pd.DataFrame(rs)

    # ---------- chosen hyperparameters (stability)
    hp = pr[pr.kind.isin(['reg', 'clf']) & (pr.shuffle_seed == -1)].drop_duplicates(['task_id', 'repeat', 'fold']).groupby(
        ['target', 'model', 'variant_label']).params.agg(lambda s: s.value_counts().head(3).to_dict()).reset_index()

    ev = run / 'evaluation'
    ev.mkdir(exist_ok=True)
    for name, df in (('continuous_pct_change', cont), ('binary_orr', binr), ('paired_comparisons', pd.DataFrame(comps)),
                     ('shuffle_control', shuf), ('reader_sensitivity', rs), ('hyperparameters', hp)):
        df.to_csv(ev / f'{name}.csv', index=False)
    print(cont[['system', 'MAE', 'MAE_lo', 'MAE_hi', 'MedAE', 'RMSE', 'bias', 'pearson', 'spearman', 'direction_agreement', 'FU_MAE_mm',
                'MAE_per_repeat_sd']].round(3).to_string(index=False))
    print(binr[['system', 'source', 'AUROC', 'AUROC_lo', 'AUROC_hi', 'AUPRC', 'balanced_accuracy', 'macro_F1', 'sensitivity', 'specificity',
                'kappa', 'Brier']].round(3).to_string(index=False))
    print(pd.DataFrame(comps).round(3).to_string(index=False))
    print(shuf.round(3).to_string(index=False))
    print(rs.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
