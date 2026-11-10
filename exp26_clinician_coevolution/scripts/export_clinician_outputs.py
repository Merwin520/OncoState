#!/usr/bin/env python3
"""Clinician-facing study tables from returned review folders. Reads NO reference data.

    python scripts/export_clinician_outputs.py [--exports exports/clinician50] [--out exports/clinician50/outputs]

Writes clinician_reviews.csv (final decision per lesion row), patient_review_summary.csv, reviewer_progress.csv and
reviewer_summary.csv. Each reviewer's hash-chained event log is verified first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.schema_mapping import load_bundle  # noqa: E402
from app.review_export import reviewer_tables  # noqa: E402

from evaluation.cohorts import cohort  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["50", "30", "10A", "10B", "10C", "10D", "10E"], default="50")
    ap.add_argument("--exports", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    c = cohort(a.cohort)
    a.exports = a.exports or c["exports"]
    out = a.out or a.exports / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    cases = load_bundle(c["bundle"])["cases"]
    rows, patients, progress = [], [], []
    for reviewer_dir in sorted(p for p in (a.exports / "events").iterdir() if p.is_dir()):
        r, p, s = reviewer_tables(reviewer_dir, cases)
        rows += r
        patients += p
        progress.append(s)
    reviews = pd.DataFrame(rows)
    per_patient = pd.DataFrame(patients)
    reviews.to_csv(out / "clinician_reviews.csv", index=False)
    per_patient.to_csv(out / "patient_review_summary.csv", index=False)
    pd.DataFrame(progress)[["reviewer_id", "patients_completed", "current_patient", "total_review_seconds"]].to_csv(
        out / "reviewer_progress.csv", index=False)
    if not per_patient.empty:
        done = per_patient[per_patient.completed]
        summary = done.groupby("reviewer_id").agg(
            patients_completed=("patient_id", "count"), median_seconds_per_patient=("patient_review_seconds", "median"),
            total_review_hours=("patient_review_seconds", lambda s: round(s.sum() / 3600, 2)),
            corrections=("n_changed", "sum"), disappeared=("n_disappeared", "sum"), new=("n_new", "sum"),
            uncertain=("n_uncertain", "sum"), median_corrections_per_patient=("n_changed", "median")).reset_index()
        summary.to_csv(out / "reviewer_summary.csv", index=False)
    print(f"{len(progress)} reviewer(s), {len(reviews)} lesion rows -> {out}")


if __name__ == "__main__":
    main()
