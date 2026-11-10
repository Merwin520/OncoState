#!/usr/bin/env python3
"""Post-hoc feasibility analysis for a risk-control gate after Hybrid CDT-R."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HYBRID = REPO / "experiments/exp1_hybrid_cdtr/results/per_patient_hybrid.csv"
V0 = REPO / "experiments/cdtr_exp1_state/results/per_patient_states.csv"
OUT, TAB, FIG = HERE / "results", HERE / "tables", HERE / "figures"
PRIMARY_ALPHA, PRIMARY_REPEAT = 0.10, 0
DELTA = 0.05
TARGETS = (0.05, 0.10, 0.15)
FRACTIONS = np.linspace(0.1, 1.0, 10)
PROVED = "PROVED_EQUIVALENT"


def cp_upper(k: int, n: int, delta: float = DELTA) -> float:
    """One-sided exact binomial upper confidence bound."""
    if n <= 0:
        return 1.0
    if not 0 <= k <= n:
        raise ValueError("require 0 <= k <= n")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def safe_percentile(cal: np.ndarray, values: np.ndarray, higher_is_safer: bool) -> np.ndarray:
    """Map values to [0,1] safety percentiles using calibration features only."""
    cal = np.asarray(cal, float)
    values = np.asarray(values, float)
    finite = cal[np.isfinite(cal)]
    if len(finite) == 0:
        return np.zeros(len(values), float)
    fill = float(np.median(finite))
    ordered = np.sort(np.where(np.isfinite(cal), cal, fill))
    x = np.where(np.isfinite(values), values, fill)
    if higher_is_safer:
        return np.searchsorted(ordered, x, side="right") / len(ordered)
    return (len(ordered) - np.searchsorted(ordered, x, side="left")) / len(ordered)


def policy_scores(cal: pd.DataFrame, target: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Return eligibility and safety score; reference outcomes are intentionally absent."""
    certified = target.hybrid_certified.to_numpy(bool)
    if method in {"confidence", "hybrid_cdtr_confidence"}:
        score = safe_percentile(cal.hyb_min_conf, target.hyb_min_conf, True)
    elif method in {"registration", "hybrid_cdtr_registration"}:
        score = safe_percentile(cal.residual_mm, target.residual_mm, False)
    elif method == "rc_cdtr":
        confidence = safe_percentile(cal.hyb_min_conf, target.hyb_min_conf, True)
        registration = safe_percentile(cal.residual_mm, target.residual_mm, False)
        complexity = safe_percentile(cal.lesion_count, target.lesion_count, False)
        score = np.minimum.reduce([confidence, registration, complexity])
    else:
        raise KeyError(method)
    gated = method.startswith("hybrid_cdtr") or method == "rc_cdtr"
    eligible = certified if gated else np.ones(len(target), bool)
    return eligible, score


def threshold_for_fraction(eligible: np.ndarray, score: np.ndarray, fraction: float) -> float:
    values = score[eligible]
    if len(values) == 0:
        return np.inf
    k = max(1, int(np.floor(fraction * len(values))))
    return float(np.sort(values)[-k])


def select_gate(cal: pd.DataFrame, method: str, epsilon: float) -> dict:
    eligible, score = policy_scores(cal, cal, method)
    wrong = cal.wrong.to_numpy(int)
    candidates = []
    adjusted_delta = DELTA / len(FRACTIONS)
    for fraction in FRACTIONS:
        threshold = threshold_for_fraction(eligible, score, float(fraction))
        accepted = eligible & (score >= threshold)
        n, k = int(accepted.sum()), int(wrong[accepted].sum())
        candidates.append({
            "fraction": float(fraction), "threshold": threshold, "n": n, "k": k,
            "risk": k / n if n else np.nan, "upper": cp_upper(k, n, adjusted_delta),
        })
    feasible = [x for x in candidates if x["upper"] <= epsilon]
    if not feasible:
        return {"selected": False, "threshold": np.inf, "cal_n": 0, "cal_k": 0,
                "cal_upper": 1.0, "candidate_count": len(candidates)}
    chosen = max(feasible, key=lambda x: (x["n"], x["threshold"]))
    return {"selected": True, "threshold": chosen["threshold"], "cal_n": chosen["n"],
            "cal_k": chosen["k"], "cal_upper": chosen["upper"],
            "candidate_count": len(candidates)}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_csv(HYBRID, dtype={"patient_id": str})
    h = h[(h.repeat == PRIMARY_REPEAT) & np.isclose(h.alpha, PRIMARY_ALPHA)].copy()
    if len(h) != 274 or not h.patient_id.is_unique:
        raise RuntimeError("expected 274 unique primary Hybrid rows")
    v = pd.read_csv(V0, dtype={"patient_id": str})
    v = v[(v.variant == "crossfit") & np.isclose(v.alpha, PRIMARY_ALPHA)].copy()
    if len(v) != 274 or not v.patient_id.is_unique:
        raise RuntimeError("expected 274 unique primary V0 rows")
    v = v.set_index("patient_id")
    d = h.set_index("patient_id").join(v[["fold", "residual"]])
    if len(d) != 274 or not d.index.is_unique or d[["fold", "residual"]].isna().all(axis=1).any():
        raise RuntimeError("Hybrid/V0 patient join was not one-to-one and complete")
    d = d.rename(columns={"residual": "residual_mm", "status": "hybrid_status"})
    d["hybrid_certified"] = d.hybrid_status.eq(PROVED).astype(int)
    d["lesion_count"] = d.n_bl + d.n_fu
    d["wrong"] = d.wrong.astype(int)
    audit_cols = [
        "fold", "hyb_response", "ref_response", "wrong", "ref_included",
        "hybrid_certified", "hybrid_status", "residual_mm", "hyb_min_conf",
        "ambiguous_lesions", "n_bl", "n_fu", "lesion_count", "n_reachable", "q",
    ]
    return d.reset_index(), d[audit_cols].reset_index()


def fixed_summary(d: pd.DataFrame) -> pd.DataFrame:
    v = pd.read_csv(V0, dtype={"patient_id": str})
    v = v[(v.variant == "crossfit") & np.isclose(v.alpha, PRIMARY_ALPHA)]
    rows = []
    for name, accept, wrong in [
        ("Original CDT-R", v.status.eq(PROVED).to_numpy(), v.wrong.to_numpy(int)),
        ("Hybrid CDT-R", d.hybrid_certified.to_numpy(bool), d.wrong.to_numpy(int)),
    ]:
        n, k = int(accept.sum()), int(wrong[accept].sum())
        rows.append({"method": name, "accepted": n, "cohort_n": len(accept),
                     "coverage": n / len(accept), "errors": k, "risk": k / n,
                     "cp_upper_95_one_sided": cp_upper(k, n)})
    return pd.DataFrame(rows)


def descriptive_curves(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = ["confidence", "registration", "hybrid_cdtr_confidence",
               "hybrid_cdtr_registration", "rc_cdtr"]
    for method in methods:
        eligible, score = policy_scores(d, d, method)
        for fraction in FRACTIONS:
            threshold = threshold_for_fraction(eligible, score, float(fraction))
            accepted = eligible & (score >= threshold)
            n, k = int(accepted.sum()), int(d.wrong.to_numpy(int)[accepted].sum())
            rows.append({"method": method, "eligible_fraction": float(fraction),
                         "threshold": threshold, "accepted": n, "coverage": n / len(d),
                         "errors": k, "risk": k / n if n else np.nan,
                         "cp_upper_95_one_sided": cp_upper(k, n)})
    return pd.DataFrame(rows)


def crossfit(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = ["confidence", "registration", "hybrid_cdtr_confidence",
               "hybrid_cdtr_registration", "rc_cdtr"]
    fold_rows, patient_rows = [], []
    for epsilon in TARGETS:
        for method in methods:
            for fold in sorted(d.fold.unique()):
                cal, test = d[d.fold != fold].copy(), d[d.fold == fold].copy()
                choice = select_gate(cal, method, epsilon)
                eligible, score = policy_scores(cal, test, method)
                accepted = (eligible & (score >= choice["threshold"])) if choice["selected"] else np.zeros(len(test), bool)
                wrong = test.wrong.to_numpy(int)
                n, k = int(accepted.sum()), int(wrong[accepted].sum())
                fold_rows.append({"epsilon": epsilon, "method": method, "fold": int(fold),
                                  **choice, "test_n": len(test), "test_accepted": n,
                                  "test_errors": k, "test_risk": k / n if n else np.nan})
                for pid, a, s, w in zip(test.patient_id, accepted, score, wrong):
                    patient_rows.append({"epsilon": epsilon, "method": method, "fold": int(fold),
                                         "patient_id": pid, "accepted": int(a),
                                         "safety_score": float(s), "wrong": int(w)})
    return pd.DataFrame(fold_rows), pd.DataFrame(patient_rows)


def aggregate_crossfit(patients: pd.DataFrame, n_cohort: int) -> pd.DataFrame:
    rows = []
    for (epsilon, method), g in patients.groupby(["epsilon", "method"], sort=False):
        n, k = int(g.accepted.sum()), int((g.accepted * g.wrong).sum())
        rows.append({"epsilon": epsilon, "method": method, "accepted": n,
                     "cohort_n": n_cohort, "coverage": n / n_cohort, "errors": k,
                     "test_risk": k / n if n else np.nan,
                     "test_cp_upper_95_one_sided": cp_upper(k, n)})
    return pd.DataFrame(rows)




def json_safe(value):
    """Convert numpy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    for directory in (OUT, TAB, FIG):
        directory.mkdir(parents=True, exist_ok=True)
    d, audit = load_data()
    fixed, curves = fixed_summary(d), descriptive_curves(d)
    folds, patients = crossfit(d)
    crossfit_summary = aggregate_crossfit(patients, len(d))
    audit.to_csv(OUT / "patient_audit.csv", index=False)
    fixed.to_csv(TAB / "fixed_methods.csv", index=False)
    curves.to_csv(TAB / "risk_coverage.csv", index=False)
    folds.to_csv(TAB / "crossfit_fold_results.csv", index=False)
    patients.to_csv(OUT / "crossfit_patient_decisions.csv", index=False)
    crossfit_summary.to_csv(TAB / "crossfit_risk_control.csv", index=False)
    inputs = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (HYBRID, V0)}
    summary = {
        "primary": {"epsilon": TARGETS[0], "delta": DELTA,
                    "candidate_policies_per_method": len(FRACTIONS),
                    "selection_delta": DELTA / len(FRACTIONS)},
        "inputs_sha256": inputs, "fixed_methods": fixed.to_dict("records"),
        "crossfit": crossfit_summary.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    print("\nFixed methods")
    print(fixed.round(4).to_string(index=False))
    print("\nCross-fitted risk-control selection")
    print(crossfit_summary.round(4).to_string(index=False))
    print(f"\nWrote {OUT}, {TAB}, and {FIG}")


if __name__ == "__main__":
    main()

