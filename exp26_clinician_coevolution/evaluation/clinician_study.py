"""OFFLINE evaluation of the clinician study against the hidden reference (never imported by the clinician app).

    python -m evaluation.clinician_study [--exports exports/clinician50] [--out exports/clinician50/results]

Unit of analysis = one V0 decision row of the review page (a prior lesion's V0 partner or "disappeared", or a follow-up
lesion V0 left unmatched / "new"); only completed patients are used. Confidence intervals resample PATIENTS
(cluster bootstrap), never lesions.

  V0 row correct        V0 decision equals the reference (same partner, or both "no partner")
  clinician row correct the clinician's final decision for that row equals the reference ("uncertain" = not correct)
  flagged               the clinician's final decision differs from V0, or is "uncertain"
  error detection       positive = V0 row incorrect; sensitivity / specificity / PPV / NPV of "flagged"
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adapters.schema_mapping import load_bundle
from app import one_page as op
from corrections.event_store import EventStore
from evaluation.cohorts import cohort
from evaluation.gt_access import load_gt
from evaluation.metrics import counts_frame, summarize
from evaluation.paired_statistics import bootstrap_f1, paired_bootstrap_f1

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/clinician_50/clinician_safe"
GT_DIR = ROOT / "secure_evaluation/cohort_ground_truth"
UNCERTAIN = "uncertain"


def _ref_partner(gt: dict) -> tuple[dict, dict]:
    bl = {a: b for a, b in gt["pairs"]}
    fu = {b: a for a, b in gt["pairs"]}
    return bl, fu


def review_rows(case: dict, gt: dict, events: list[dict]) -> list[dict]:
    """One row per V0 decision with V0 / clinician / reference outcome."""
    decided = op.decisions(events)
    res = op.resolve(case, decided)
    ref_bl, ref_fu = _ref_partner(gt)
    final_bl = dict(res["target"])
    final_fu = {f: b for b, f in res["target"].items() if f}
    for f, b in res["fu_link"].items():
        final_bl[b], final_fu[f] = f, b
    rows = []
    v0 = res["v0"]
    for b in res["bl_ids"]:
        d = decided.get(f"bl:{b}", {"decision": "keep", "implicit": True})
        clin = UNCERTAIN if d["decision"] == UNCERTAIN else final_bl.get(b)
        rows.append({"patient_id": case["patient_id"], "row": f"bl:{b}", "row_type": "prior_lesion", "v0": v0.get(b),
                     "clinician": clin, "reference": ref_bl.get(b), "explicit": not d.get("implicit", True)})
    for f in sorted({l["lesion_id"] for l in case["followup_lesions"]} - set(v0.values()), key=op.natural):
        d = decided.get(f"fu:{f}", {"decision": "new", "implicit": True})
        clin = UNCERTAIN if (f in res["unclaimed"] and d["decision"] == UNCERTAIN) else final_fu.get(f)
        rows.append({"patient_id": case["patient_id"], "row": f"fu:{f}", "row_type": "v0_new_followup", "v0": None,
                     "clinician": clin, "reference": ref_fu.get(f), "explicit": not d.get("implicit", True)})
    for r in rows:
        r["v0_correct"] = r["v0"] == r["reference"]
        r["clinician_correct"] = r["clinician"] != UNCERTAIN and r["clinician"] == r["reference"]
        r["uncertain"] = r["clinician"] == UNCERTAIN
        r["flagged"] = r["uncertain"] or r["clinician"] != r["v0"]
    return rows


def _ratio(df: pd.DataFrame, num: str, den: str) -> float:
    d = df[den].sum()
    return float(df[num].sum() / d) if d else float("nan")


def cluster_ci(rows: pd.DataFrame, num_mask, den_mask, n_boot: int, seed: int) -> dict:
    """Proportion num/den with a patient-level (cluster) bootstrap 95% CI."""
    per = rows.assign(num=num_mask(rows) & den_mask(rows), den=den_mask(rows)).groupby("patient_id")[["num", "den"]].sum()
    point = _ratio(per, "num", "den")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(per))
    vals = [_ratio(per.iloc[rng.choice(idx, len(idx))], "num", "den") for _ in range(n_boot)] if len(per) else []
    lo, hi = (np.nanpercentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return {"value": point, "ci_lo": float(lo), "ci_hi": float(hi), "numerator": int(per.num.sum()), "denominator": int(per.den.sum()),
            "n_patients": int((per.den > 0).sum())}


METRICS = {
    # A. accuracy
    "accuracy_overall": (lambda d: d.clinician_correct, lambda d: d.v0 == d.v0),
    "accuracy_on_v0_correct": (lambda d: d.clinician_correct, lambda d: d.v0_correct),
    "accuracy_on_v0_incorrect": (lambda d: d.clinician_correct, lambda d: ~d.v0_correct),
    "v0_accuracy_reference": (lambda d: d.v0_correct, lambda d: d.v0 == d.v0),
    # B. error detection (positive = V0 incorrect)
    "sensitivity_error_detection": (lambda d: d.flagged, lambda d: ~d.v0_correct),
    "specificity_retain_correct": (lambda d: ~d.flagged, lambda d: d.v0_correct),
    "ppv_flag": (lambda d: ~d.v0_correct, lambda d: d.flagged),
    "npv_flag": (lambda d: d.v0_correct, lambda d: ~d.flagged),
    # C. correction accuracy among flagged V0 errors
    "flagged_errors_corrected_to_reference": (lambda d: d.clinician_correct, lambda d: ~d.v0_correct & d.flagged),
    "flagged_errors_changed_to_other_incorrect": (lambda d: ~d.clinician_correct & ~d.uncertain, lambda d: ~d.v0_correct & d.flagged),
    "flagged_errors_marked_uncertain": (lambda d: d.uncertain, lambda d: ~d.v0_correct & d.flagged),
    # D. harm to correct V0 decisions
    "v0_correct_retained": (lambda d: ~d.flagged, lambda d: d.v0_correct),
    "v0_correct_incorrectly_changed": (lambda d: d.flagged & ~d.uncertain, lambda d: d.v0_correct),
    "v0_correct_marked_uncertain": (lambda d: d.uncertain, lambda d: d.v0_correct),
    "uncertain_rate": (lambda d: d.uncertain, lambda d: d.v0 == d.v0),
}


def accuracy_table(rows: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    out = []
    groups = [("ALL", rows)] + list(rows.groupby("reviewer_id")) if rows.reviewer_id.nunique() > 1 else list(rows.groupby("reviewer_id"))
    for reviewer, df in groups:
        df = df.assign(v0=df.v0.fillna("∅"))
        for name, (num, den) in METRICS.items():
            out.append({"reviewer_id": reviewer, "metric": name, **cluster_ci(df, num, den, n_boot, seed)})
    return pd.DataFrame(out)


def state_f1(cases, gt, finals: dict, pids, n_boot, seed) -> list[dict]:
    """Lesion-matching F1 of V0 vs the clinician-corrected states on the 50 review patients (patient bootstrap)."""
    v0 = {p: op.v0_pairs(cases[p]).items() for p in pids}
    d0 = counts_frame(cases, {p: list(v0[p]) for p in pids}, gt, pids)
    d1 = counts_frame(cases, {p: finals[p] for p in pids}, gt, pids)
    f0, f1 = bootstrap_f1(d0, n_boot, seed), bootstrap_f1(d1, n_boot, seed)
    dl = paired_bootstrap_f1(d1, d0, n_boot, seed)
    return [{"state": "V0", **{k: v for k, v in summarize(d0).items()}, "f1_ci_lo": f0[1], "f1_ci_hi": f0[2]},
            {"state": "clinician_corrected", **{k: v for k, v in summarize(d1).items()}, "f1_ci_lo": f1[1], "f1_ci_hi": f1[2],
             "delta_f1_vs_v0": dl["delta"], "delta_ci_lo": dl["ci_lo"], "delta_ci_hi": dl["ci_hi"]}]


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(bool), b.astype(bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def agreement_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r1, r2 in itertools.combinations(sorted(rows.reviewer_id.unique()), 2):
        a = rows[rows.reviewer_id == r1].set_index(["patient_id", "row"])
        b = rows[rows.reviewer_id == r2].set_index(["patient_id", "row"])
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        a, b = a.loc[common], b.loc[common]
        out.append({"reviewer_a": r1, "reviewer_b": r2, "n_rows": len(common), "n_patients": common.get_level_values(0).nunique(),
                    "percent_agreement_keep_vs_change": float((a.flagged == b.flagged).mean()),
                    "cohen_kappa_keep_vs_change": cohen_kappa(a.flagged, b.flagged),
                    "percent_agreement_final_lesion_match": float((a.clinician.fillna("∅") == b.clinician.fillna("∅")).mean()),
                    "both_correct": float((a.clinician_correct & b.clinician_correct).mean())})
    return pd.DataFrame(out)


def workload_table(rows: pd.DataFrame, progress: pd.DataFrame) -> pd.DataFrame:
    out = []
    for reviewer, p in progress.groupby("reviewer_id"):
        r = rows[rows.reviewer_id == reviewer]
        n_rows = r.groupby("patient_id").size()
        sec = p.set_index("patient_id").seconds
        changes = r[r.flagged & ~r.uncertain].groupby("patient_id").size().reindex(sec.index, fill_value=0)
        out.append({"reviewer_id": reviewer, "patients_completed": len(p), "total_review_hours": round(sec.sum() / 3600, 2),
                    "median_seconds_per_patient": float(sec.median()),
                    "median_seconds_per_lesion_row": float((sec / n_rows.reindex(sec.index)).median()),
                    "median_corrections_per_patient": float(changes.median()), "uncertain_rate": float(r.uncertain.mean()),
                    "explicit_edit_rate": float(r.explicit.mean())})
    return pd.DataFrame(out)


def collect(exports: Path, cases: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    gt = load_gt(GT_DIR)
    rows, progress, finals = [], [], {}
    for reviewer_dir in sorted(p for p in (exports / "events").iterdir() if p.is_dir()):
        store = EventStore(reviewer_dir, sqlite_mirror=False)
        store.verify()
        status = json.loads((reviewer_dir / "case_status.json").read_text(encoding="utf-8")) if (reviewer_dir / "case_status.json").exists() else {}
        prog = json.loads((reviewer_dir / "patient_progress.json").read_text(encoding="utf-8")) if (reviewer_dir / "patient_progress.json").exists() else {}
        events = store.all()
        for pid in sorted(p for p, s in status.items() if s.get("status") == "Completed"):
            evs = [e for e in events if e["patient_id"] == pid]
            rows += [{**r, "reviewer_id": reviewer_dir.name, "simulated": any(e["correction_source"] != "clinician" for e in evs)}
                     for r in review_rows(cases[pid], gt[pid], evs)]
            progress.append({"reviewer_id": reviewer_dir.name, "patient_id": pid, "seconds": prog.get(pid, {}).get("seconds", np.nan)})
            finals[(reviewer_dir.name, pid)] = op.resolve(cases[pid], op.decisions(evs))["pairs"]
    return pd.DataFrame(rows), pd.DataFrame(progress), {"gt": gt, "finals": finals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["50", "30", "10A", "10B", "10C", "10D", "10E"], default="50")
    ap.add_argument("--exports", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()
    global BUNDLE, GT_DIR
    c = cohort(a.cohort)
    BUNDLE, GT_DIR = c["bundle"], c["secure"]
    a.exports = a.exports or c["exports"]
    out = a.out or a.exports / "results"
    out.mkdir(parents=True, exist_ok=True)
    cases = load_bundle(BUNDLE)["cases"]
    rows, progress, extra = collect(a.exports, cases)
    if rows.empty:
        raise SystemExit("no completed patients in the returned review folders")
    simulated = bool(rows.simulated.any())
    rows.to_csv(out / "clinician_review_rows_with_reference.csv", index=False)
    acc = accuracy_table(rows, a.n_boot, a.seed).assign(data_source="SIMULATED" if simulated else "clinician")
    acc.to_csv(out / "clinician_accuracy.csv", index=False)
    agreement_table(rows).to_csv(out / "interreader_agreement.csv", index=False)
    workload_table(rows, progress).to_csv(out / "clinician_workload.csv", index=False)
    f1_rows = []
    for reviewer in sorted(rows.reviewer_id.unique()):
        pids = sorted(rows[rows.reviewer_id == reviewer].patient_id.unique())
        finals = {p: extra["finals"][(reviewer, p)] for p in pids}
        f1_rows += [{"reviewer_id": reviewer, **r} for r in state_f1(cases, extra["gt"], finals, pids, a.n_boot, a.seed)]
    pd.DataFrame(f1_rows).to_csv(out / "corrected_state_matching.csv", index=False)
    print(f"{'SIMULATED ' if simulated else ''}clinician study evaluation: {rows.reviewer_id.nunique()} reviewer(s), "
          f"{rows.patient_id.nunique()} patients, {len(rows)} V0 decision rows -> {out}")


if __name__ == "__main__":
    main()
