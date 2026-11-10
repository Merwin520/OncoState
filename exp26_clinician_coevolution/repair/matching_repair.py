"""Lesion matchers.

CentroidHungarianMatcher  frozen V0 = exact replica of the repository's MatchingNode 'registered_centroid_hungarian'
                          (experiments/exp1_agentic_longitudinal/exp22b_methods.py::hungarian_pairs): 30 mm gate,
                          unmatched cost gate/2 on both sides, one-to-one Hungarian.
PairwiseLogisticMatcher   repair candidate: standardised logistic regression P(same lesion | pair features), then the same
                          one-to-one Hungarian assignment on cost -logit(p) with unmatched cost -logit(threshold)/2.
Both are fully described by a JSON dict (no pickle), so every model version is human-readable and hashable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .matching_features import candidate_pairs, distance_matrix


def _assign(cost: np.ndarray, unmatched: float) -> list[tuple[int, int]]:
    """Repository assignment: dummy rows/columns with cost unmatched/... so pairs costlier than 2*unmatched stay unmatched."""
    n, m = cost.shape
    big = 1e6
    c = np.full((n + m, n + m), big)
    c[:n, :m] = cost
    c[:n, m:] = unmatched
    c[n:, :m] = unmatched
    c[n:, m:] = 0
    rr, cc = linear_sum_assignment(c)
    return [(i, j) for i, j in zip(rr, cc) if i < n and j < m and cost[i, j] < big]


class CentroidHungarianMatcher:
    kind = "registered_centroid_hungarian"

    def __init__(self, gate_mm: float = 30.0):
        self.gate_mm = gate_mm

    def to_dict(self) -> dict:
        return {"type": self.kind, "gate_mm": self.gate_mm}

    def predict(self, case: dict) -> dict:
        bids, fids, d = distance_matrix(case)
        if not bids or not fids:
            return {"pairs": [], "confidence": {}, "distance": {}}
        cost = np.where(d <= self.gate_mm, d, 1e6)
        pairs = [(bids[i], fids[j]) for i, j in _assign(cost, self.gate_mm / 2) if d[i, j] <= self.gate_mm]
        conf = {(a, b): float(max(0.0, 1 - d[bids.index(a), fids.index(b)] / self.gate_mm)) for a, b in pairs}
        dist = {(a, b): float(d[bids.index(a), fids.index(b)]) for a, b in pairs}
        return {"pairs": pairs, "confidence": conf, "distance": dist}


class PairwiseLogisticMatcher:
    kind = "pairwise_logistic"

    def __init__(self, features: list[str], mean, scale, coef, intercept: float, threshold: float = 0.5,
                 max_distance_mm: float = 100.0, v0_gate_mm: float = 30.0, training: dict | None = None):
        self.features = list(features)
        self.mean, self.scale, self.coef = (np.asarray(x, float) for x in (mean, scale, coef))
        self.intercept, self.threshold = float(intercept), float(threshold)
        self.max_distance_mm, self.v0_gate_mm = float(max_distance_mm), float(v0_gate_mm)
        self.training = training or {}

    def to_dict(self) -> dict:
        return {"type": self.kind, "features": self.features, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept, "threshold": self.threshold,
                "max_distance_mm": self.max_distance_mm, "v0_gate_mm": self.v0_gate_mm, "training": self.training}

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        z = ((X[self.features].to_numpy(float) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, case: dict) -> dict:
        v0 = set(CentroidHungarianMatcher(self.v0_gate_mm).predict(case)["pairs"])
        cand = candidate_pairs(case, v0, self.max_distance_mm)
        if cand.empty:
            return {"pairs": [], "confidence": {}, "distance": {}}
        p = np.clip(self.proba(cand), 1e-6, 1 - 1e-6)
        bids = [l["lesion_id"] for l in case["baseline_lesions"]]
        fids = [l["lesion_id"] for l in case["followup_lesions"]]
        cost = np.full((len(bids), len(fids)), 1e6)
        prob = {}
        for (a, b, pr) in zip(cand.baseline_lesion_id, cand.followup_lesion_id, p):
            prob[(a, b)] = float(pr)
            if pr >= self.threshold:
                cost[bids.index(a), fids.index(b)] = -np.log(pr / (1 - pr))
        unmatched = -np.log(self.threshold / (1 - self.threshold)) / 2 if self.threshold != 0.5 else 0.0
        pairs = [(bids[i], fids[j]) for i, j in _assign(cost, unmatched)]
        dist = dict(zip(zip(cand.baseline_lesion_id, cand.followup_lesion_id), cand.dist_mm))
        return {"pairs": pairs, "confidence": {k: prob[k] for k in pairs}, "distance": {k: float(dist[k]) for k in pairs}}


def matcher_from_dict(d: dict):
    if d["type"] == CentroidHungarianMatcher.kind:
        return CentroidHungarianMatcher(d["gate_mm"])
    if d["type"] == PairwiseLogisticMatcher.kind:
        return PairwiseLogisticMatcher(**{k: d[k] for k in ("features", "mean", "scale", "coef", "intercept", "threshold",
                                                             "max_distance_mm", "v0_gate_mm")}, training=d.get("training"))
    raise ValueError(f"unknown matcher type {d['type']}")
