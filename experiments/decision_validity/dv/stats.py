"""Patient-level statistics: paired bootstrap, Clopper-Pearson, Holm, fractional-tie top-k error."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta


def cp(k, n):
    if n == 0:
        return (np.nan, np.nan)
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))


def boot_idx(n, B, seed):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def paired(fn, idx):
    """fn(ix) -> scalar for a resample ix of patient positions; returns point, (lo, hi), two-sided bootstrap p for 0."""
    pt = fn(np.arange(idx.shape[1]))
    bs = np.array([fn(ix) for ix in idx], float)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return float(pt), (float(lo), float(hi)), float(p)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def topk_error(score, wrong, k):
    """Expected error of the k highest-score patients, ties broken uniformly at random (fractional)."""
    score, wrong = np.asarray(score, float), np.asarray(wrong, float)
    if k <= 0:
        return np.nan
    o = np.argsort(-score, kind='stable'); s, w = score[o], wrong[o]; t = s[k - 1]
    above, tied = s > t, s == t
    return float((w[above].sum() + (k - above.sum()) / tied.sum() * w[tied].sum()) / k)
