"""Patient-level statistics. Lesions are never resampled independently of their patient."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from .metrics import summarize


def bootstrap_metric(df: pd.DataFrame, metric: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = summarize(df)[metric]
    n = len(df)
    vals = [summarize(df.iloc[rng.integers(0, n, n)])[metric] for _ in range(n_boot)]
    return point, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _f1_from_sums(tp, pp, gp):
    den = pp + gp
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)


def paired_bootstrap_f1(df_a: pd.DataFrame, df_b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Vectorised patient-level paired bootstrap of pooled matching F1 (A - B)."""
    a = df_a.set_index("patient_id").sort_index()
    b = df_b.set_index("patient_id").loc[a.index]
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, (n_boot, n))
    fa = _f1_from_sums(a.tp.to_numpy()[idx].sum(1), a.pred_pairs.to_numpy()[idx].sum(1), a.gt_pairs.to_numpy()[idx].sum(1))
    fb = _f1_from_sums(b.tp.to_numpy()[idx].sum(1), b.pred_pairs.to_numpy()[idx].sum(1), b.gt_pairs.to_numpy()[idx].sum(1))
    d = fa - fb
    point = float(_f1_from_sums(a.tp.sum(), a.pred_pairs.sum(), a.gt_pairs.sum()) - _f1_from_sums(b.tp.sum(), b.pred_pairs.sum(), b.gt_pairs.sum()))
    p = float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))
    return {"delta": point, "ci_lo": float(np.nanpercentile(d, 2.5)), "ci_hi": float(np.nanpercentile(d, 97.5)), "p_boot": p, "n_patients": n}


def bootstrap_f1(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.integers(0, n, (n_boot, n))
    f = _f1_from_sums(df.tp.to_numpy()[idx].sum(1), df.pred_pairs.to_numpy()[idx].sum(1), df.gt_pairs.to_numpy()[idx].sum(1))
    return float(_f1_from_sums(df.tp.sum(), df.pred_pairs.sum(), df.gt_pairs.sum())), float(np.nanpercentile(f, 2.5)), float(np.nanpercentile(f, 97.5))


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, n_boot: int, seed: int) -> dict:
    """Delta = metric(A) - metric(B) on the same patients; resampling patients jointly."""
    if metric == "matching_f1":
        return paired_bootstrap_f1(df_a, df_b, n_boot, seed)
    a = df_a.set_index("patient_id").sort_index()
    b = df_b.set_index("patient_id").loc[a.index]
    rng = np.random.default_rng(seed)
    n = len(a)
    point = summarize(a.reset_index())[metric] - summarize(b.reset_index())[metric]
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d.append(summarize(a.iloc[idx].reset_index())[metric] - summarize(b.iloc[idx].reset_index())[metric])
    d = np.asarray(d)
    p = float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))) if np.isfinite(d).any() else float("nan")
    return {"delta": point, "ci_lo": float(np.nanpercentile(d, 2.5)), "ci_hi": float(np.nanpercentile(d, 97.5)), "p_boot": p, "n_patients": n}


def mcnemar(decisions_a: pd.Series, decisions_b: pd.Series) -> dict:
    """Exact McNemar on paired binary decisions (e.g. GT pair recovered by A vs by B). Lesions are clustered in patients:
    report alongside the patient-level bootstrap, never alone."""
    b01 = int(((~decisions_a) & decisions_b).sum())
    b10 = int((decisions_a & (~decisions_b)).sum())
    p = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    return {"a_only_correct": b10, "b_only_correct": b01, "p_exact": float(p)}


def wilcoxon_patient(f_a: pd.Series, f_b: pd.Series) -> dict:
    d = (f_a - f_b).dropna()
    d = d[d != 0]
    if len(d) < 5:
        return {"n_nonzero": int(len(d)), "p": float("nan")}
    return {"n_nonzero": int(len(d)), "p": float(wilcoxon(d).pvalue)}


def per_patient_f1(df: pd.DataFrame) -> pd.Series:
    den = df.pred_pairs + df.gt_pairs
    return pd.Series(np.where(den > 0, 2 * df.tp / den.where(den > 0, 1), np.nan), index=df.patient_id)


def benjamini_hochberg(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        k = n - rank + 1
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj.tolist()
