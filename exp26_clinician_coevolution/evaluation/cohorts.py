"""Clinician-study cohorts (server side). The 30-case reader study is a subset of the 50-case cohort, and the 10A / 10B
packages are disjoint subsets of the 30-case cohort; validation-gate and
future-unseen patients are identical for both (they exclude all 50 patients), so results stay comparable."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "50": {"bundle": ROOT / "data/clinician_50/clinician_safe", "secure": ROOT / "secure_evaluation/cohort_ground_truth",
           "config": ROOT / "configs/ui_clinician50.yaml", "exports": ROOT / "exports/clinician50"},
    "30": {"bundle": ROOT / "data/clinician_30/clinician_safe", "secure": ROOT / "secure_evaluation/cohort_ground_truth_30",
           "config": ROOT / "configs/ui_clinician30.yaml", "exports": ROOT / "exports/clinician30"},
    # two disjoint 10-case packages split from the 30-case cohort (scripts/select_cohort10.py); each is shared as its own ZIP
    **{n: {"bundle": ROOT / f"data/clinician_{n}/clinician_safe", "secure": ROOT / f"secure_evaluation/cohort_ground_truth_{n}",
           "config": ROOT / f"configs/ui_clinician{n}.yaml", "exports": ROOT / f"exports/clinician{n}"} for n in ("10A", "10B", "10C", "10D", "10E")},
}
ALL_REVIEW_PATIENTS = ROOT / "secure_evaluation/cohort_ground_truth/selected_ids.json"  # superset; excluded from gate/future


def cohort(name: str | None) -> dict:
    return COHORTS[str(name or "50")]
