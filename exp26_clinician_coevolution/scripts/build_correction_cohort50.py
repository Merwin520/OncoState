#!/usr/bin/env python3
"""Build the real, correction-enriched 50-patient clinician cohort.

Hidden Experiment 2.2 correspondence is read only while this server-side build
runs.  The clinician bundle receives frozen V0 predictions but no reference
labels, correctness flags, selection strata, or hidden paths.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from build_real_pilot import _assert_safe, _convert_exam, _json, _sha256  # noqa: E402
from repair.matching_repair import CentroidHungarianMatcher  # noqa: E402
from state.provenance import sha256_json  # noqa: E402


# Neutral locked order: strata were shuffled together before this list was frozen.
# Error/correct membership is computed from hidden reference and written only to
# secure_evaluation/cohort_selection_truth.csv.
SELECTED_IDS = [
    "2a79ea27c2", "2618045a3a", "a597e50502", "84eb13cfed", "7eabe3a164",
    "2a452c50d7", "82aa4b0af3", "db8e1af0cb", "0e65972dce", "bf8229696f",
    "7f100b7b36", "31b7b72368", "903ce9225f", "13c022b0b3", "a0a080f42e",
    "335f535208", "3def184ad8", "0e51011a4c", "7f6ffaa6bb", "0266e33d3f",
    "9cfdf10e8f", "bf97f24695", "bbf14f4f9b", "045117b0e0", "060ad92489",
    "46ba9f2a69", "091d584fce", "c9e1074f5b", "979d472a84", "1d7f7abc18",
    "5737c6ec2e", "8f03ad02c6", "c4258e0f53", "eda80a3d5b", "738cccd4fd",
    "3636638817", "a8c88a0055", "577ef1154f", "65ded5353c", "73f7634ab3",
    "502e4a1693", "9dcb88e013", "758874998f", "4c5bde74a8", "a8baa56554",
    "a27ef1c79c", "26e359e838", "bd686fd640", "13fe9d8431", "e4a6222cdb",
]

TRUTH_FIELDS = [
    "patient_id", "selection_stratum", "has_v0_error", "baseline_lesions", "followup_lesions",
    "total_review_decisions", "correct_v0_matches", "incorrect_v0_matches",
    "correct_disappeared_decisions", "incorrect_disappeared_decisions",
    "correct_new_decisions", "incorrect_new_decisions", "mean_predicted_pair_distance_mm",
    "median_lesion_diameter_mm", "reference_change_classes",
]


def _audit(case: dict, gt: dict, prediction: dict) -> dict:
    predicted = set(map(tuple, prediction["pairs"]))
    reference = set(map(tuple, gt["pairs"]))
    matched_bl = {a for a, _ in predicted}
    matched_fu = {b for _, b in predicted}
    unmatched_bl = {x["lesion_id"] for x in case["baseline_lesions"]} - matched_bl
    unmatched_fu = {x["lesion_id"] for x in case["followup_lesions"]} - matched_fu
    correct_dis = sum(gt["baseline_class"].get(a) == "disappearing" for a in unmatched_bl)
    incorrect_dis = len(unmatched_bl) - correct_dis
    correct_new = sum(gt["followup_class"].get(b) == "newly_appearing" for b in unmatched_fu)
    incorrect_new = len(unmatched_fu) - correct_new
    incorrect_pairs = len(predicted - reference)
    distances = [float(x) for x in prediction.get("distance", {}).values()]
    diameters = [float(x["diameter_mm"]) for x in case["baseline_lesions"] + case["followup_lesions"]]
    has_error = bool(incorrect_pairs or incorrect_dis or incorrect_new)
    return {
        "patient_id": case["patient_id"],
        "selection_stratum": "v0_error" if has_error else "v0_reference_correct",
        "has_v0_error": has_error,
        "baseline_lesions": len(case["baseline_lesions"]),
        "followup_lesions": len(case["followup_lesions"]),
        "total_review_decisions": len(predicted) + len(unmatched_bl) + len(unmatched_fu),
        "correct_v0_matches": len(predicted & reference),
        "incorrect_v0_matches": incorrect_pairs,
        "correct_disappeared_decisions": correct_dis,
        "incorrect_disappeared_decisions": incorrect_dis,
        "correct_new_decisions": correct_new,
        "incorrect_new_decisions": incorrect_new,
        "mean_predicted_pair_distance_mm": round(float(np.mean(distances)), 3) if distances else "",
        "median_lesion_diameter_mm": round(float(np.median(diameters)), 3) if diameters else "",
        "reference_change_classes": "|".join(sorted(set(gt.get("change", {}).values()))),
    }


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _reuse_pilot(pid: str, pilot: Path, clinician: Path) -> dict | None:
    source_case = pilot / "cases" / f"{pid}.json"
    if not source_case.is_file():
        return None
    case = _json(source_case)
    for side in ("baseline", "followup"):
        volume = Path(case[f"{side}_exam"]["image_ref"])
        _link_or_copy(pilot / volume, clinician / volume)
        for lesion in case[f"{side}_lesions"]:
            crop = Path(lesion["crop_ref"])
            _link_or_copy(pilot / crop, clinician / crop)
    transform = Path(case["registration"]["transform_ref"])
    _link_or_copy(pilot / transform, clinician / transform)
    return case


def _convert_one(pid: str, args: argparse.Namespace, clinician: Path) -> tuple[dict, dict]:
    import SimpleITK as sitk

    case = copy.deepcopy(_json(args.source_bundle / "cases" / f"{pid}.json"))
    stats = {}
    for side in ("baseline", "followup"):
        exam_key, lesions_key = f"{side}_exam", f"{side}_lesions"
        image_path = args.image_root / case[exam_key]["image_ref"]
        mask_path = args.exp22_output / "cut_A" / f"{pid}_{side}.nii.gz"
        out_path = clinician / "volumes" / f"{pid}_{side}.npz"
        info, exam_stats = _convert_exam(
            sitk, image_path, mask_path, case[lesions_key], side, out_path, clinician,
        )
        case[exam_key].update(info)
        stats[side] = exam_stats
    transform_source = args.exp22_output / "reg" / f"{pid}.tfm"
    transform_target = clinician / "transforms" / f"{pid}.tfm"
    shutil.copyfile(transform_source, transform_target)
    case["registration"]["transform_ref"] = f"transforms/{pid}.tfm"
    return case, stats


def _freeze_prediction(case: dict, prediction: dict) -> None:
    case["v0_prediction"] = {
        "model_version": "V0",
        "matcher": "registered_centroid_hungarian",
        "gate_mm": 30.0,
        "pairs": [list(pair) for pair in prediction["pairs"]],
        "confidence": {"|".join(pair): value for pair, value in prediction["confidence"].items()},
        "centroid_distance_mm": {"|".join(pair): value for pair, value in prediction["distance"].items()},
    }
    sb, sf = case["baseline_exam"]["shape_zyx"], case["followup_exam"]["shape_zyx"]
    spb, spf = case["baseline_exam"]["spacing_zyx_mm"], case["followup_exam"]["spacing_zyx_mm"]
    synced = sb == sf and np.allclose(spb, spf, atol=0.05)
    case["navigation"] = {
        "synchronized_scrolling_valid": bool(synced),
        "basis": "identical array shape and voxel spacing" if synced else "independent",
    }


def build(args: argparse.Namespace) -> None:
    global SELECTED_IDS
    if args.selected_ids:
        SELECTED_IDS = json.loads(args.selected_ids.read_text(encoding="utf-8"))
    args.source_bundle = args.source_bundle.resolve()
    args.image_root = args.image_root.resolve()
    args.exp22_output = args.exp22_output.resolve()
    clinician = args.clinician.resolve()
    truth_path = args.selection_truth.resolve()
    if clinician.exists() and any(clinician.iterdir()):
        raise SystemExit(f"{clinician} is not empty; remove a failed build deliberately before rebuilding")
    clinician.mkdir(parents=True, exist_ok=True)
    for subdir in ("cases", "volumes", "crops", "transforms"):
        (clinician / subdir).mkdir(parents=True, exist_ok=True)

    source_manifest = _json(args.source_bundle / "manifest.json")
    cohort = {p["patient_id"]: p["cohort"] for p in source_manifest["patients"]}
    if sum(value == "locked" for value in cohort.values()) != 274:
        raise ValueError("source is not the 274-patient locked Experiment 2.2 cohort")
    if len(SELECTED_IDS) != 50 or len(set(SELECTED_IDS)) != 50:
        raise ValueError("selection must contain 50 unique patients")
    if any(cohort.get(pid) != "locked" for pid in SELECTED_IDS):
        raise ValueError("selection contains a patient outside the locked source cohort")

    matcher = CentroidHungarianMatcher(30.0)
    source_cases = {pid: _json(args.source_bundle / "cases" / f"{pid}.json") for pid in SELECTED_IDS}
    predictions = {pid: matcher.predict(source_cases[pid]) for pid in SELECTED_IDS}
    audits = {
        pid: _audit(source_cases[pid], _json(args.source_bundle / "gt_hidden" / f"{pid}.json"), predictions[pid])
        for pid in SELECTED_IDS
    }
    n_error = sum(row["has_v0_error"] for row in audits.values())
    n_correct = len(audits) - n_error
    incorrect_pairs = sum(row["incorrect_v0_matches"] for row in audits.values())
    if (n_error, n_correct) != (30, 20):
        raise ValueError(f"hidden-reference composition is {n_error}/{n_correct}, expected 30/20")
    if incorrect_pairs < 40:
        raise ValueError(f"only {incorrect_pairs} incorrect V0 pairs; minimum is 40")

    pilot = args.reuse_pilot.resolve()
    built: dict[str, dict] = {}
    conversion_stats: dict[str, dict] = {}
    pending = []
    for pid in SELECTED_IDS:
        reused = _reuse_pilot(pid, pilot, clinician) if pilot.is_dir() else None
        if reused is not None:
            built[pid] = reused
            conversion_stats[pid] = {"reused_validated_real_pilot": True}
            print(f"[{len(built):02d}/50] reused {pid}", flush=True)
        else:
            pending.append(pid)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_convert_one, pid, args, clinician): pid for pid in pending}
        for future in as_completed(futures):
            pid = futures[future]
            built[pid], conversion_stats[pid] = future.result()
            print(f"[{len(built):02d}/50] converted {pid}", flush=True)

    cases = []
    for pid in SELECTED_IDS:
        case = built[pid]
        _freeze_prediction(case, predictions[pid])
        _assert_safe(case, f"case:{pid}")
        (clinician / "cases" / f"{pid}.json").write_text(json.dumps(case, indent=1, sort_keys=True))
        cases.append(case)

    manifest = {
        "cohort": args.cohort_name,
        "data_mode": "real",
        "model_version": "V0",
        "n_patients": 50,
        "patients": [{"patient_id": pid, "cohort": "locked"} for pid in SELECTED_IDS],
        "cases_sha256": sha256_json(cases),
        "volume_format": "compressed NPZ (int16 HU CT + uint16 predicted label map)",
        "source_cohort_size": 274,
    }
    _assert_safe(manifest, "manifest")
    (clinician / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))

    truth_path.parent.mkdir(parents=True, exist_ok=True)
    with truth_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRUTH_FIELDS)
        writer.writeheader()
        writer.writerows(audits[pid] for pid in SELECTED_IDS)

    build_audit = {
        "patient_id": [],
        "clinician_manifest_sha256": _sha256(clinician / "manifest.json"),
        "volume_count": len(list((clinician / "volumes").glob("*.npz"))),
        "case_count": len(list((clinician / "cases").glob("*.json"))),
        "conversion": conversion_stats,
    }
    # This file is provenance only: it contains no correctness or selection labels.
    (clinician / "build_provenance.json").write_text(json.dumps(build_audit, indent=1, sort_keys=True))

    totals = {
        key: sum(int(row[key]) for row in audits.values())
        for key in (
            "total_review_decisions", "incorrect_v0_matches", "correct_v0_matches",
            "incorrect_new_decisions", "incorrect_disappeared_decisions",
        )
    }
    print(f"patients with >=1 V0 error: {n_error}")
    print(f"patients with all V0 matches correct: {n_correct}")
    print(f"total lesion decisions: {totals['total_review_decisions']}")
    print(f"incorrect V0 lesion matches: {totals['incorrect_v0_matches']}")
    print(f"correct V0 lesion matches: {totals['correct_v0_matches']}")
    print(f"new-lesion errors: {totals['incorrect_new_decisions']}")
    print(f"disappeared-lesion errors: {totals['incorrect_disappeared_decisions']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, default=ROOT / "data/real/exp22_longitudinal_ct")
    parser.add_argument("--clinician", type=Path, default=ROOT / "data/real_correction_cohort/clinician")
    parser.add_argument("--selection-truth", type=Path, default=ROOT / "secure_evaluation/cohort_selection_truth.csv")
    parser.add_argument("--reuse-pilot", type=Path, default=ROOT / "data/real_pilot/clinician")
    parser.add_argument("--image-root", type=Path, default=ROOT.parents[1] / "Longitudinal-CT")
    parser.add_argument(
        "--exp22-output", type=Path,
        default=ROOT.parent / "experiments/exp1_agentic_longitudinal/outputs/exp22_baselines/exp22_regen",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--selected-ids", type=Path, default=None,
                        help="JSON list of 50 patient IDs (e.g. secure_evaluation/cohort_ground_truth/selected_ids.json)")
    parser.add_argument("--cohort-name", default="exp22_locked_real_correction50")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
