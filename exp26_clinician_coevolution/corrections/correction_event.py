"""Construction of immutable correction events.

clinician_event(...) requires a ui_action_token minted by the clinician UI for a concrete button press.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import jsonschema

from state.provenance import now_iso, sha256_json
from state.schemas import GT_FIELD_NAMES

SCHEMA = json.loads((Path(__file__).parent / "schemas" / "correction_event.schema.json").read_text(encoding="utf-8"))

ACTION_TO_TYPE = {"accept": "lesion_match", "correct": "lesion_match", "disappeared": "disappeared_lesion",
                  "new_lesion": "new_lesion", "uncertain": "uncertain", "segmentation_ok": "segmentation_quality",
                  "segmentation_incorrect": "segmentation_quality", "measurement": "measurement", "status": "lesion_status"}


def new_event_id() -> str:
    return "E" + secrets.token_hex(8)


def mint_ui_action_token() -> str:
    """Called by the UI inside a button callback; the token travels with the resulting event."""
    return "ui-" + secrets.token_hex(8)


def _base(*, patient_id, batch_id, action, model_version, state_version_before, review_item, baseline_lesion_id,
          original_followup_lesion_id, corrected_followup_lesion_id=None, original_prediction=None, corrected_value=None,
          original_confidence=None, registration_quality=None, centroid_distance=None, review_time_seconds=None,
          clinician_comment=None, data_mode="synthetic", measurement_side=None, session_id=None) -> dict:
    ev = {
        "event_id": new_event_id(), "patient_id": patient_id, "batch_id": batch_id, "timestamp": now_iso(),
        "event_type": ACTION_TO_TYPE[action], "action": action, "data_mode": data_mode, "model_version": model_version,
        "state_version_before": int(state_version_before), "state_version_after": int(state_version_before) + 1,
        "review_item": review_item, "baseline_lesion_id": baseline_lesion_id,
        "original_followup_lesion_id": original_followup_lesion_id, "corrected_followup_lesion_id": corrected_followup_lesion_id,
        "original_prediction": original_prediction, "corrected_value": corrected_value,
        "original_confidence": original_confidence, "registration_quality": registration_quality,
        "centroid_distance": centroid_distance, "centroid_distance_mm": centroid_distance,
        "review_time_seconds": review_time_seconds, "clinician_comment": clinician_comment,
        "uncertain_flag": action == "uncertain", "measurement_side": measurement_side, "session_id": session_id,
    }
    leaked = GT_FIELD_NAMES & set(ev)
    if leaked:
        raise ValueError(f"GT fields must not enter correction events: {sorted(leaked)}")
    return ev


def clinician_event(*, clinician_id: str, ui_action_token: str, **kw) -> dict:
    if not clinician_id or clinician_id.upper() == "ORACLE":
        raise ValueError("clinician events need a study clinician ID (never ORACLE)")
    if not ui_action_token or not str(ui_action_token).startswith("ui-"):
        raise PermissionError("clinician events can only be created from a UI action (missing ui_action_token)")
    ev = _base(**kw)
    ev.update(clinician_id=clinician_id, correction_source="clinician", ui_action_token=ui_action_token)
    return ev



def seal(ev: dict, prev_sha: str | None) -> dict:
    """Attach the hash chain and validate. Returns a new dict."""
    out = dict(ev, prev_event_sha256=prev_sha)
    out["event_sha256"] = sha256_json({k: v for k, v in out.items() if k != "event_sha256"})
    jsonschema.validate(out, SCHEMA)
    return out


def is_correction(ev: dict) -> bool:
    """A 'verified correction' changes the model's answer (accept / uncertain / segmentation_ok do not)."""
    if ev["action"] == "correct":
        return ev["corrected_followup_lesion_id"] != ev["original_followup_lesion_id"]
    if ev["action"] == "disappeared":
        return ev["original_followup_lesion_id"] is not None
    if ev["action"] == "new_lesion":
        return ev["baseline_lesion_id"] is not None
    return ev["action"] in ("measurement", "segmentation_incorrect", "status")
