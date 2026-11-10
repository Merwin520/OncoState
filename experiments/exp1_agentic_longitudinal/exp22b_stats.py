"""Experiment 2.2 baselines: metrics, patient-level bootstrap (5,000), paired differences, consistency, efficiency, failure matrix.

Bootstrap unit = patient (never lesion). All methods share the same bootstrap resamples, so differences are paired.
Generic-agent statistics are computed from per-patient counts averaged over its repeats (expected metric under repeated sampling);
the per-repeat spread is reported separately as mean +- SD.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

import exp22b_common as C

BASE = C.BASE
NB, SEED = 5000, 20260921
METRICS = ['match_precision', 'match_recall', 'match_f1', 'status_accuracy', 'status_weighted_f1', 'status_macro_f1', 'status_kappa',
           'change_accuracy_conditional', 'change_kappa_conditional', 'change_accuracy_unconditional',
           'burden_accuracy', 'burden_balanced_accuracy', 'burden_kappa', 'task_success_rate']


def vec(ev):
    """Per-patient sufficient statistics as one flat vector."""
    return np.concatenate([[ev['tp'], ev['pred_pairs'], ev['gt_pers'], ev['chg_cond_n'], ev['chg_cond_ok'], ev['chg_unc_n'], ev['chg_unc_ok']],
                           ev['conf'].ravel(), ev['chg_conf'].ravel(), ev['burden_conf'].ravel()]).astype(float)


N_HEAD = 7
def unpack(S):
    """S: (..., D) summed sufficient statistics -> dict of arrays."""
    return dict(tp=S[..., 0], pp=S[..., 1], gp=S[..., 2], chg_n=S[..., 3], chg_ok=S[..., 4], unc_n=S[..., 5], unc_ok=S[..., 6],
                conf=S[..., N_HEAD:N_HEAD + 12].reshape(S.shape[:-1] + (3, 4)), chg=S[..., N_HEAD + 12:N_HEAD + 21].reshape(S.shape[:-1] + (3, 3)),
                bur=S[..., N_HEAD + 21:N_HEAD + 33].reshape(S.shape[:-1] + (3, 4)))


def kappa_square(c):
    n = c.sum((-2, -1)); nn = np.where(n == 0, np.nan, n)
    po = np.trace(c, axis1=-2, axis2=-1) / nn; pe = (c.sum(-1) * c.sum(-2)).sum(-1) / nn ** 2
    return (po - pe) / (1 - pe)


def metrics_from_sums(S, success):
    u = unpack(S); out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        out['match_precision'] = u['tp'] / u['pp']; out['match_recall'] = u['tp'] / u['gp']; out['match_f1'] = 2 * u['tp'] / (u['pp'] + u['gp'])
        conf = u['conf']; n = conf.sum((-2, -1)); out['status_accuracy'] = np.trace(conf[..., :3], axis1=-2, axis2=-1) / n
        tp = np.stack([conf[..., i, i] for i in range(3)], -1); fn = conf.sum(-1) - tp; fp = conf[..., :3].sum(-2) - tp
        f1 = 2 * tp / (2 * tp + fp + fn); w = conf.sum(-1)
        out['status_weighted_f1'] = np.nansum(f1 * w, -1) / w.sum(-1); out['status_macro_f1'] = np.nanmean(f1, -1)
        pad = np.zeros(conf.shape[:-2] + (4, 4)); pad[..., :3, :] = conf; out['status_kappa'] = kappa_square(pad)
        out['change_accuracy_conditional'] = u['chg_ok'] / u['chg_n']; out['change_kappa_conditional'] = kappa_square(u['chg']); out['change_accuracy_unconditional'] = u['unc_ok'] / u['unc_n']
        b = u['bur']; nb = b.sum((-2, -1)); out['burden_accuracy'] = np.trace(b[..., :3], axis1=-2, axis2=-1) / nb
        rec = np.stack([b[..., i, i] / b[..., i, :].sum(-1) for i in range(3)], -1); out['burden_balanced_accuracy'] = np.nanmean(rec, -1)
        padb = np.zeros(b.shape[:-2] + (4, 4)); padb[..., :3, :] = b; out['burden_kappa'] = kappa_square(padb)
        out['task_success_rate'] = success
    return out


def bh(ps):
    ps = np.asarray(ps, float); n = len(ps); o = np.argsort(ps); q = np.empty(n); prev = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        prev = min(prev, ps[i] * n / rank); q[i] = prev
    return q


def analyse(per_method, success_by_method, patients, extra=None):
    """per_method: {method: {pid: vector}}; success_by_method: {method: {pid: float in [0,1]}}."""
    rng = np.random.default_rng(SEED); n = len(patients)
    idx = rng.integers(0, n, size=(NB, n))
    M = {m: np.stack([per_method[m][p] for p in patients]) for m in per_method}
    Sc = {m: np.array([success_by_method[m][p] for p in patients], float) for m in per_method}
    point, boot = {}, {}
    for m in M:
        point[m] = {k: float(v) for k, v in metrics_from_sums(M[m].sum(0), Sc[m].mean()).items()}
        boot[m] = {k: np.asarray(v) for k, v in metrics_from_sums(M[m][idx].sum(1), Sc[m][idx].mean(1)).items()}
    rows = []
    for m in M:
        for k in METRICS:
            v = boot[m][k]; rows.append(dict(method=m, metric=k, estimate=point[m][k], ci95_lo=np.nanpercentile(v, 2.5), ci95_hi=np.nanpercentile(v, 97.5), n_patients=n))
    return pd.DataFrame(rows), point, boot, idx, M, Sc


def paired(point, boot, ref='Full OncoAgent'):
    rows = []
    for m in point:
        if m == ref: continue
        for k in METRICS:
            d = boot[ref][k] - boot[m][k]
            rows.append(dict(comparison=f'{ref} - {m}', metric=k, difference=point[ref][k] - point[m][k], ci95_lo=np.nanpercentile(d, 2.5), ci95_hi=np.nanpercentile(d, 97.5)))
    return pd.DataFrame(rows)


def mcnemar_rows(patients, a, b, name, label):
    """Exact McNemar test on patient-level binary outcomes (a, b: {pid: 0/1/None})."""
    x = np.array([a[p] for p in patients], float); y = np.array([b[p] for p in patients], float)
    ok = ~np.isnan(x) & ~np.isnan(y); x, y = x[ok] >= .5, y[ok] >= .5
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    return dict(comparison=name, endpoint=label, only_first_correct=n10, only_second_correct=n01, p_value=binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0)
