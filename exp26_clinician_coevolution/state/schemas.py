"""Schemas shared by the Experiment 2.6 package."""
from __future__ import annotations

PACKAGE_SCHEMA_VERSION = "exp26-state-1.0"

# Annotation keys that can never occur in clinician cases, states, payloads, or events.
GT_FIELD_NAMES = frozenset({
    "gt", "gt_pairs", "gt_id", "gt_status", "gt_change", "gt_burden",
    "ground_truth", "ground_truth_mask", "ground_truth_match",
    "reference", "reference_pairs", "reference_match", "reference_mask", "reference_lesion",
    "true_match", "is_correct", "match_correct", "pair_correct", "oracle_label", "evaluation_label",
})

EVENT_TYPES = ("lesion_match", "lesion_status", "measurement", "segmentation_quality", "new_lesion", "disappeared_lesion", "uncertain")
CORRECTION_SOURCES = ("clinician", "oracle")
DATA_MODES = ("real", "synthetic")
REVIEW_STATUSES = ("Not reviewed", "In progress", "Completed", "Flagged", "Excluded")
LESION_STATUSES = ("persistent", "disappearing", "newly_appearing")
CHANGE_CLASSES = ("enlarged", "stable", "shrunk")
BURDEN_DIRECTIONS = ("increase", "decrease", "no_change")

LESION_SCHEMA = {
    "type": "object",
    "required": ["lesion_id", "centroid_mm", "volume_mm3", "diameter_mm"],
    "properties": {
        "lesion_id": {"type": "string"},
        "organ": {"type": ["string", "null"]},
        "centroid_mm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "centroid_registered_mm": {"type": ["array", "null"]},
        "volume_mm3": {"type": "number"},
        "diameter_mm": {"type": "number"},
        "mask_ref": {"type": ["string", "null"]},
        "crop_ref": {"type": ["string", "null"]},
        "segmentation_confidence": {"type": ["number", "null"]},
    },
}

PATIENT_STATE_SCHEMA = {
    "type": "object",
    "required": ["patient_id", "state_version", "schema_version", "source_model_version", "baseline_exam", "followup_exam",
                 "baseline_lesions", "followup_lesions", "correspondences", "patient_level", "provenance", "correction_history"],
    "properties": {
        "patient_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]+$"},
        "state_version": {"type": "integer", "minimum": 0},
        "parent_state_version": {"type": ["integer", "null"]},
        "schema_version": {"type": "string"},
        "data_mode": {"enum": list(DATA_MODES)},
        "source_model_version": {"type": "string"},
        "baseline_exam": {"type": "object"},
        "followup_exam": {"type": "object"},
        "baseline_lesions": {"type": "array", "items": LESION_SCHEMA},
        "followup_lesions": {"type": "array", "items": LESION_SCHEMA},
        "correspondences": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["baseline_lesion_id", "followup_lesion_id", "status", "source"],
                "properties": {
                    "baseline_lesion_id": {"type": ["string", "null"]},
                    "followup_lesion_id": {"type": ["string", "null"]},
                    "status": {"enum": list(LESION_STATUSES)},
                    "confidence": {"type": ["number", "null"]},
                    "centroid_distance_mm": {"type": ["number", "null"]},
                    "change_class": {"type": ["string", "null"]},
                    "source": {"enum": ["model", "clinician", "oracle"]},
                    "event_id": {"type": ["string", "null"]},
                },
            },
        },
        "patient_level": {"type": "object"},
        "provenance": {"type": "object"},
        "correction_history": {"type": "array"},
    },
    "not": {"anyOf": [{"required": [key]} for key in sorted(GT_FIELD_NAMES)]},
}
