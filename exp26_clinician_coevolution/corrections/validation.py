"""Semantic validation of a correction event against the patient state it refers to."""
from __future__ import annotations


class InvalidCorrection(ValueError):
    pass


def validate_against_state(event: dict, state: dict) -> None:
    if event["patient_id"] != state["patient_id"]:
        raise InvalidCorrection("patient mismatch")
    if event["state_version_before"] != state["state_version"]:
        raise InvalidCorrection(f"stale state: event refers to v{event['state_version_before']}, current is v{state['state_version']}")
    bl = {l["lesion_id"] for l in state["baseline_lesions"]}
    fu = {l["lesion_id"] for l in state["followup_lesions"]}
    a, b0, b1 = event["baseline_lesion_id"], event["original_followup_lesion_id"], event["corrected_followup_lesion_id"]
    if a is not None and a not in bl:
        raise InvalidCorrection(f"unknown baseline lesion {a}")
    for b in (b0, b1):
        if b is not None and b not in fu:
            raise InvalidCorrection(f"unknown follow-up lesion {b}")
    if event["action"] == "correct" and b1 is None:
        raise InvalidCorrection("'Incorrect match' needs a selected follow-up candidate (or use Disappeared)")
    if event["action"] == "correct" and b1 == b0:
        raise InvalidCorrection("corrected follow-up lesion equals the original prediction; use Accept")
