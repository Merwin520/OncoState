"""Longitudinal patient state: construction from a case record and a model prediction.

The patient state is what the clinician sees and corrects. It never contains reference (GT) data, and it stores only
references to pixel data (mask_ref / crop_ref), never the pixels themselves.
"""
from __future__ import annotations

import copy

import jsonschema
import numpy as np

from .provenance import now_iso
from .schemas import PACKAGE_SCHEMA_VERSION, PATIENT_STATE_SCHEMA

CHANGE_THRESHOLD_PCT = 20.0  # +-20 % volume change, the frozen Experiment 2.2 definition


def change_class(vol_bl: float, vol_fu: float) -> str | None:
    if not vol_bl:
        return None
    pct = (vol_fu - vol_bl) / vol_bl * 100
    return "enlarged" if pct > CHANGE_THRESHOLD_PCT else "shrunk" if pct < -CHANGE_THRESHOLD_PCT else "stable"


def burden_direction(total_bl: float, total_fu: float) -> str:
    return "increase" if total_fu > total_bl else "decrease" if total_fu < total_bl else "no_change"


def correspondences_from_pairs(case: dict, pairs: list[tuple[str, str]], confidence: dict | None = None,
                               distance: dict | None = None, source: str = "model") -> list[dict]:
    """Full correspondence list: matched pairs + unmatched baseline (disappearing) + unmatched follow-up (new)."""
    bl = {l["lesion_id"]: l for l in case["baseline_lesions"]}
    fu = {l["lesion_id"]: l for l in case["followup_lesions"]}
    mb, mf = {a for a, _ in pairs}, {b for _, b in pairs}
    out = []
    for a, b in sorted(pairs):
        out.append({"baseline_lesion_id": a, "followup_lesion_id": b, "status": "persistent",
                    "confidence": None if confidence is None else float(confidence.get((a, b), np.nan)),
                    "centroid_distance_mm": None if distance is None else float(distance.get((a, b), np.nan)),
                    "change_class": change_class(bl[a]["volume_mm3"], fu[b]["volume_mm3"]), "source": source, "event_id": None})
    for a in sorted(set(bl) - mb):
        out.append({"baseline_lesion_id": a, "followup_lesion_id": None, "status": "disappearing", "confidence": None,
                    "centroid_distance_mm": None, "change_class": None, "source": source, "event_id": None})
    for b in sorted(set(fu) - mf):
        out.append({"baseline_lesion_id": None, "followup_lesion_id": b, "status": "newly_appearing", "confidence": None,
                    "centroid_distance_mm": None, "change_class": None, "source": source, "event_id": None})
    for c in out:  # JSON has no NaN
        for k in ("confidence", "centroid_distance_mm"):
            if c[k] is not None and np.isnan(c[k]):
                c[k] = None
    return out


def patient_level(case: dict) -> dict:
    tb = float(sum(l["volume_mm3"] for l in case["baseline_lesions"]))
    tf = float(sum(l["volume_mm3"] for l in case["followup_lesions"]))
    return {"tumour_burden_baseline_mm3": tb, "tumour_burden_followup_mm3": tf, "burden_direction": burden_direction(tb, tf),
            "registration_quality": case.get("registration", {}).get("quality", {}), "qc_flags": list(case.get("qc_flags", []))}


def build_state(case: dict, pairs: list[tuple[str, str]], model_version: str, confidence: dict | None = None,
                distance: dict | None = None, tool_versions: dict | None = None) -> dict:
    """State v0 = the model's prediction for this patient."""
    state = {
        "patient_id": case["patient_id"], "state_version": 0, "parent_state_version": None,
        "schema_version": PACKAGE_SCHEMA_VERSION, "data_mode": case.get("data_mode", "synthetic"),
        "source_model_version": model_version,
        "baseline_exam": copy.deepcopy(case["baseline_exam"]), "followup_exam": copy.deepcopy(case["followup_exam"]),
        "baseline_lesions": copy.deepcopy(case["baseline_lesions"]), "followup_lesions": copy.deepcopy(case["followup_lesions"]),
        "correspondences": correspondences_from_pairs(case, pairs, confidence, distance),
        "patient_level": patient_level(case),
        "provenance": {"created": now_iso(), "created_by": f"model:{model_version}", "case_source": case.get("source", "unknown"),
                       "tool_versions": tool_versions or {}, "registration_ref": case.get("registration", {}).get("transform_ref")},
        "correction_history": [],
    }
    validate_state(state)
    return state


def validate_state(state: dict) -> None:
    jsonschema.validate(state, PATIENT_STATE_SCHEMA)


def predicted_pairs(state: dict) -> list[tuple[str, str]]:
    return [(c["baseline_lesion_id"], c["followup_lesion_id"]) for c in state["correspondences"] if c["status"] == "persistent"]
