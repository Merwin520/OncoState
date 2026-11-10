#!/usr/bin/env python3
"""Build the fixed 10-patient Experiment 2.6 real-case clinician pilot.

The clinician bundle contains only de-identified CT voxels, predicted masks,
predicted lesion metadata, frozen V0 predictions, and frozen transforms.
Evaluation annotations are copied to ``secure_evaluation/real_pilot`` and are
never placed below the clinician bundle.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repair.matching_repair import CentroidHungarianMatcher  # noqa: E402
from state.provenance import sha256_json  # noqa: E402


# Locked before export. The order intentionally alternates simpler and harder cases.
PILOT_IDS = [
    "091d584fce",  # single lesion, lower-confidence correct pair
    "979d472a84",  # single lesion, high-confidence correct pair
    "1d7f7abc18",  # single lesion, difficult unmatched V0 state
    "13c022b0b3",  # small multi-lesion, correct pairs
    "045117b0e0",  # small multi-lesion, correct pairs
    "bbf14f4f9b",  # small multi-lesion, mixed correct/incorrect pairs
    "26e359e838",  # small multi-lesion, incorrect V0 pairs
    "9dcb88e013",  # moderate multi-lesion, mixed case
    "eda80a3d5b",  # moderate multi-lesion, mixed difficult case
    "0e51011a4c",  # larger multi-lesion, mixed difficult case
]

FORBIDDEN_KEYS = {
    "ground_truth", "reference", "reference_pairs", "reference_match", "reference_mask",
    "gt", "gt_status", "gt_change", "gt_pairs", "oracle_label", "true_match", "is_correct",
    "match_correct", "pair_correct",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _assert_safe(obj, where: str) -> None:
    if isinstance(obj, dict):
        bad = FORBIDDEN_KEYS.intersection(k.lower() for k in obj)
        if bad:
            raise ValueError(f"forbidden clinician key(s) in {where}: {sorted(bad)}")
        for key, value in obj.items():
            _assert_safe(value, f"{where}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _assert_safe(value, f"{where}[{i}]")


def _window(image_hu: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    lo = center - width / 2.0
    return np.clip((image_hu.astype(np.float32) - lo) * (255.0 / width), 0, 255).astype(np.uint8)


def _crop(image_hu: np.ndarray, labels: np.ndarray, label: int, spacing_zyx: np.ndarray,
          out_path: Path, fov_mm: float = 112.0, size: int = 256) -> None:
    points = np.argwhere(labels == label)
    if not len(points):
        raise ValueError(f"label {label} missing from {out_path.name}")
    z, y, x = np.rint(points.mean(axis=0)).astype(int)
    half_y = max(24, int(round(fov_mm / max(float(spacing_zyx[1]), 1e-6) / 2)))
    half_x = max(24, int(round(fov_mm / max(float(spacing_zyx[2]), 1e-6) / 2)))
    y0, y1 = max(0, y - half_y), min(image_hu.shape[1], y + half_y)
    x0, x1 = max(0, x - half_x), min(image_hu.shape[2], x + half_x)
    gray = _window(image_hu[z, y0:y1, x0:x1])
    mask = labels[z, y0:y1, x0:x1] == label
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    rgb[mask] = rgb[mask] * 0.55 + np.array([59, 91, 165], dtype=np.float32) * 0.45
    square_side = max(rgb.shape[:2])
    canvas = np.zeros((square_side, square_side, 3), dtype=np.uint8)
    oy, ox = (square_side - rgb.shape[0]) // 2, (square_side - rgb.shape[1]) // 2
    canvas[oy:oy + rgb.shape[0], ox:ox + rgb.shape[1]] = np.clip(rgb, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).resize((size, size), Image.Resampling.BILINEAR).save(out_path, optimize=True)


def _convert_exam(sitk, image_path: Path, mask_path: Path, lesions: list[dict], side: str,
                  out_path: Path, clinician_root: Path) -> tuple[dict, dict]:
    started = time.perf_counter()
    image_itk = sitk.ReadImage(str(image_path))
    mask_itk = sitk.ReadImage(str(mask_path))
    image = sitk.GetArrayFromImage(image_itk)
    labels = sitk.GetArrayFromImage(mask_itk)
    if image.shape != labels.shape:
        raise ValueError(f"CT/mask shape mismatch: {image_path.name}: {image.shape} vs {labels.shape}")
    image_hu = np.rint(np.clip(image, -32768, 32767)).astype(np.int16)
    labels = labels.astype(np.uint16, copy=False)
    ids = [lesion["lesion_id"] for lesion in lesions]
    centroid_zyx = []
    for lesion in lesions:
        label = int(lesion["lesion_id"][1:])
        points = np.argwhere(labels == label)
        if not len(points):
            raise ValueError(f"predicted {lesion['lesion_id']} absent from {mask_path}")
        centroid_zyx.append(points.mean(axis=0).tolist())
        crop_rel = f"crops/{image_path.name[:10]}/{side}_{lesion['lesion_id']}.png"
        _crop(image_hu, labels, label, np.asarray(image_itk.GetSpacing()[::-1]), clinician_root / crop_rel)
        lesion["crop_ref"] = crop_rel
        lesion["mask_ref"] = f"volumes/{out_path.name}#label={label}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        image_hu=image_hu,
        label=labels,
        lesion_ids=np.asarray(ids, dtype="U16"),
        centroid_zyx=np.asarray(centroid_zyx, dtype=np.float32).reshape((-1, 3)),
        spacing_zyx=np.asarray(image_itk.GetSpacing()[::-1], dtype=np.float32),
        origin_xyz=np.asarray(image_itk.GetOrigin(), dtype=np.float64),
        direction=np.asarray(image_itk.GetDirection(), dtype=np.float64),
    )
    info = {
        "image_ref": f"volumes/{out_path.name}",
        "format": "compressed_npz_int16_hu",
        "shape_zyx": list(image_hu.shape),
        "spacing_zyx_mm": [float(x) for x in image_itk.GetSpacing()[::-1]],
        "window_default": {"center_hu": 40, "width_hu": 400},
    }
    stats = {
        "source_sha256": _sha256(image_path),
        "mask_source_sha256": _sha256(mask_path),
        "output_sha256": _sha256(out_path),
        "conversion_seconds": round(time.perf_counter() - started, 3),
    }
    return info, stats


def build(args: argparse.Namespace) -> None:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("SimpleITK is required; use the exp26 environment") from exc

    source_bundle = args.source_bundle.resolve()
    clinician = args.clinician.resolve()
    secure = args.secure_evaluation.resolve()
    if (clinician / "manifest.json").exists() and not args.force:
        raise SystemExit(f"{clinician} already exists; use --force only for a deliberate rebuild")
    clinician.mkdir(parents=True, exist_ok=True)
    secure.mkdir(parents=True, exist_ok=True)
    for subdir in ("cases", "volumes", "crops", "transforms"):
        (clinician / subdir).mkdir(parents=True, exist_ok=True)
    (secure / "annotations").mkdir(parents=True, exist_ok=True)

    source_manifest = _json(source_bundle / "manifest.json")
    cohort = {p["patient_id"]: p["cohort"] for p in source_manifest["patients"]}
    if sum(v == "locked" for v in cohort.values()) != 274:
        raise ValueError("source is not the 274-patient locked Experiment 2.2 cohort")

    matcher = CentroidHungarianMatcher(30.0)
    cases, audit = [], []
    for pid in PILOT_IDS:
        if cohort.get(pid) != "locked":
            raise ValueError(f"pilot patient {pid} is not in the locked cohort")
        case = copy.deepcopy(_json(source_bundle / "cases" / f"{pid}.json"))
        annotation = _json(source_bundle / "gt_hidden" / f"{pid}.json")
        prediction = matcher.predict(case)
        predicted_pairs = set(map(tuple, prediction["pairs"]))
        annotated_pairs = set(map(tuple, annotation["pairs"]))

        exam_stats = {}
        for side in ("baseline", "followup"):
            exam_key, lesions_key = f"{side}_exam", f"{side}_lesions"
            image_path = args.image_root / case[exam_key]["image_ref"]
            mask_path = args.exp22_output / "cut_A" / f"{pid}_{side}.nii.gz"
            out_path = clinician / "volumes" / f"{pid}_{side}.npz"
            info, stats = _convert_exam(sitk, image_path, mask_path, case[lesions_key], side, out_path, clinician)
            case[exam_key].update(info)
            exam_stats[side] = stats

        transform_source = args.exp22_output / "reg" / f"{pid}.tfm"
        transform_target = clinician / "transforms" / f"{pid}.tfm"
        shutil.copyfile(transform_source, transform_target)
        case["registration"]["transform_ref"] = f"transforms/{pid}.tfm"
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
        sync_valid = sb == sf and np.allclose(spb, spf, atol=0.05)
        case["navigation"] = {
            "synchronized_scrolling_valid": bool(sync_valid),
            "basis": "identical array shape and voxel spacing" if sync_valid else "independent",
        }
        _assert_safe(case, f"case:{pid}")
        (clinician / "cases" / f"{pid}.json").write_text(json.dumps(case, indent=1, sort_keys=True))
        (secure / "annotations" / f"{pid}.json").write_text(json.dumps(annotation, indent=1, sort_keys=True))
        cases.append(case)
        audit.append({
            "patient_id": pid,
            "baseline_lesions": len(case["baseline_lesions"]),
            "followup_lesions": len(case["followup_lesions"]),
            "v0_pairs": len(predicted_pairs),
            "v0_correct_pairs": len(predicted_pairs & annotated_pairs),
            "v0_incorrect_pairs": len(predicted_pairs - annotated_pairs),
            "v0_missed_pairs": len(annotated_pairs - predicted_pairs),
            "exam_conversion": exam_stats,
        })

    manifest = {
        "cohort": "exp22_locked_real_pilot",
        "data_mode": "real",
        "model_version": "V0",
        "n_patients": len(cases),
        "patients": [{"patient_id": p, "cohort": "locked"} for p in PILOT_IDS],
        "cases_sha256": sha256_json(cases),
        "volume_format": "compressed NPZ (int16 HU CT + uint16 predicted label map)",
        "source_cohort_size": 274,
    }
    _assert_safe(manifest, "manifest")
    (clinician / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    secure_manifest = {
        "purpose": "offline evaluation only; never read by the clinician application",
        "pilot_patient_ids": PILOT_IDS,
        "source_cohort_size": 274,
        "selection_audit": audit,
    }
    (secure / "manifest.json").write_text(json.dumps(secure_manifest, indent=1, sort_keys=True))
    print(json.dumps({"clinician_bundle": str(clinician), "secure_evaluation": str(secure), "patients": len(cases)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, default=ROOT / "data/real/exp22_longitudinal_ct")
    parser.add_argument("--clinician", type=Path, default=ROOT / "data/real_pilot/clinician")
    parser.add_argument("--secure-evaluation", type=Path, default=ROOT / "secure_evaluation/real_pilot")
    parser.add_argument("--image-root", type=Path, default=ROOT.parents[1] / "Longitudinal-CT")
    parser.add_argument("--exp22-output", type=Path,
                        default=ROOT.parent / "experiments/exp1_agentic_longitudinal/outputs/exp22_baselines/exp22_regen")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
