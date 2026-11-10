"""Pairwise features for longitudinal lesion matching (computed from predictions only; never from GT).

Every baseline lesion is paired with every follow-up lesion within `max_distance_mm` of its registered centroid.
Features (configurable subset in configs/experiment.yaml):
  dist_mm               registered centroid distance (baseline centroid mapped into follow-up space)
  dist_over_diam        dist_mm / max(diameter_bl, diameter_fu)
  log_vol_ratio         log(volume_fu / volume_bl)
  abs_log_vol_ratio     |log_vol_ratio|
  rel_diam_diff         |d_fu - d_bl| / max(d_bl, d_fu)
  log_vol_bl            log baseline volume (mL)
  rank_fu               rank of this follow-up lesion among the baseline lesion's candidates (1 = nearest)
  rank_bl               rank of this baseline lesion among the follow-up lesion's candidates
  ratio_second_fu       dist_mm / distance to the 2nd-nearest follow-up candidate (1 if none)
  ratio_second_bl       dist_mm / distance to the 2nd-nearest baseline candidate (1 if none)
  reg_residual_mm       patient-level, GT-free registration residual (median nearest-neighbour distance of all
                        registered baseline centroids to follow-up centroids)
  n_bl, n_fu            lesion counts
  organ_match           1 / 0 when organ labels exist, 0.5 when unknown (no organ labels in Longitudinal-CT)
  v0_match              1 if the frozen V0 matcher paired these two lesions
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALL_FEATURES = ["dist_mm", "dist_over_diam", "log_vol_ratio", "abs_log_vol_ratio", "rel_diam_diff", "log_vol_bl", "rank_fu", "rank_bl",
                "ratio_second_fu", "ratio_second_bl", "reg_residual_mm", "n_bl", "n_fu", "organ_match", "v0_match"]


def registered_centroids(case: dict) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    bl, fu = case["baseline_lesions"], case["followup_lesions"]
    bids = [l["lesion_id"] for l in bl]
    fids = [l["lesion_id"] for l in fu]
    bc = np.array([l.get("centroid_registered_mm") or l["centroid_mm"] for l in bl], float).reshape(-1, 3)
    fc = np.array([l["centroid_mm"] for l in fu], float).reshape(-1, 3)
    return bids, bc, fids, fc


def distance_matrix(case: dict) -> tuple[list[str], list[str], np.ndarray]:
    bids, bc, fids, fc = registered_centroids(case)
    if not bids or not fids:
        return bids, fids, np.zeros((len(bids), len(fids)))
    return bids, fids, np.linalg.norm(bc[:, None, :] - fc[None, :, :], axis=2)


def registration_residual(case: dict) -> float:
    bids, fids, d = distance_matrix(case)
    return float(np.median(d.min(axis=1))) if d.size else float("nan")


def candidate_pairs(case: dict, v0_pairs: set | None = None, max_distance_mm: float = 100.0) -> pd.DataFrame:
    bids, fids, d = distance_matrix(case)
    if not bids or not fids:
        return pd.DataFrame(columns=["patient_id", "baseline_lesion_id", "followup_lesion_id"] + ALL_FEATURES)
    bl = {l["lesion_id"]: l for l in case["baseline_lesions"]}
    fu = {l["lesion_id"]: l for l in case["followup_lesions"]}
    res = case.get("registration", {}).get("quality", {}).get("residual_mm")
    res = registration_residual(case) if res is None else res
    rank_fu = d.argsort(axis=1).argsort(axis=1) + 1
    rank_bl = d.argsort(axis=0).argsort(axis=0) + 1
    srt_r, srt_c = np.sort(d, axis=1), np.sort(d, axis=0)
    rows = []
    for i, a in enumerate(bids):
        for j, b in enumerate(fids):
            if d[i, j] > max_distance_mm:
                continue
            vb, vf = max(bl[a]["volume_mm3"], 1e-3), max(fu[b]["volume_mm3"], 1e-3)
            db, df_ = bl[a]["diameter_mm"], fu[b]["diameter_mm"]
            second_fu = srt_r[i, 1] if d.shape[1] > 1 else np.nan
            second_bl = srt_c[1, j] if d.shape[0] > 1 else np.nan
            ob, of = bl[a].get("organ"), fu[b].get("organ")
            rows.append({
                "patient_id": case["patient_id"], "baseline_lesion_id": a, "followup_lesion_id": b,
                "dist_mm": d[i, j], "dist_over_diam": d[i, j] / max(db, df_, 1e-3),
                "log_vol_ratio": np.log(vf / vb), "abs_log_vol_ratio": abs(np.log(vf / vb)),
                "rel_diam_diff": abs(df_ - db) / max(db, df_, 1e-3), "log_vol_bl": np.log(vb / 1000),
                "rank_fu": rank_fu[i, j], "rank_bl": rank_bl[i, j],
                "ratio_second_fu": d[i, j] / second_fu if second_fu and np.isfinite(second_fu) and second_fu > 0 else 1.0,
                "ratio_second_bl": d[i, j] / second_bl if second_bl and np.isfinite(second_bl) and second_bl > 0 else 1.0,
                "reg_residual_mm": res, "n_bl": len(bids), "n_fu": len(fids),
                "organ_match": 0.5 if not ob or not of else float(ob == of),
                "v0_match": float(v0_pairs is not None and (a, b) in v0_pairs),
            })
    return pd.DataFrame(rows)
