"""Immutable patient-state versions.

    states/<patient_id>/v0000.json   original model prediction (never modified)
    states/<patient_id>/v0001.json   after correction event E1
    ...

A clinician correction creates a NEW version of the CURRENT PATIENT's state. It never touches model parameters
(those live in repair/version_manager.py) and never rewrites an existing version file.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .patient_state import change_class, validate_state
from .provenance import now_iso, sha256_json


class ImmutableVersionError(RuntimeError):
    pass


class StateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, pid: str) -> Path:
        return self.root / pid

    def versions(self, pid: str) -> list[int]:
        d = self._dir(pid)
        return sorted(int(p.stem[1:]) for p in d.glob("v*.json")) if d.exists() else []

    def latest_version(self, pid: str) -> int | None:
        v = self.versions(pid)
        return v[-1] if v else None

    def load(self, pid: str, version: int | None = None) -> dict:
        v = self.latest_version(pid) if version is None else version
        if v is None:
            raise FileNotFoundError(pid)
        return json.loads((self._dir(pid) / f"v{v:04d}.json").read_text(encoding="utf-8"))

    def write(self, state: dict) -> Path:
        """Write-once: refuses to overwrite an existing version (also enforced by O_EXCL)."""
        validate_state(state)
        d = self._dir(state["patient_id"])
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"v{state['state_version']:04d}.json"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError as e:
            raise ImmutableVersionError(f"{path} exists; state versions are immutable") from e
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, sort_keys=True)
        return path

    def init_patient(self, state_v0: dict) -> dict:
        if self.versions(state_v0["patient_id"]):
            return self.load(state_v0["patient_id"])  # resume: v0 already exists
        if state_v0["state_version"] != 0:
            raise ValueError("initial state must be version 0")
        self.write(state_v0)
        return state_v0


def apply_event(state: dict, event: dict) -> dict:
    """Return a NEW state (version + 1) that reflects one correction event. The input state is not modified."""
    new = copy.deepcopy(state)
    new["parent_state_version"] = state["state_version"]
    new["state_version"] = state["state_version"] + 1
    corr = new["correspondences"]
    bl = {l["lesion_id"]: l for l in new["baseline_lesions"]}
    fu = {l["lesion_id"]: l for l in new["followup_lesions"]}
    a, b_new = event.get("baseline_lesion_id"), event.get("corrected_followup_lesion_id")
    src = event["correction_source"]
    et, action = event["event_type"], event.get("action")

    def drop(pred):
        corr[:] = [c for c in corr if not pred(c)]

    if et == "lesion_match" and action == "correct" and a is not None and b_new is not None:
        # a <-> b_new becomes the verified pair; any previous pairing of a or b_new is removed.
        drop(lambda c: c["baseline_lesion_id"] == a or c["followup_lesion_id"] == b_new)
        old_b = event.get("original_followup_lesion_id")
        corr.append({"baseline_lesion_id": a, "followup_lesion_id": b_new, "status": "persistent", "confidence": None,
                     "centroid_distance_mm": event.get("centroid_distance_mm"),
                     "change_class": change_class(bl[a]["volume_mm3"], fu[b_new]["volume_mm3"]), "source": src, "event_id": event["event_id"],
                     "verified_by_event": event["event_id"]})
        # Every lesion released by the re-pairing (a's old partner, or the baseline previously paired with b_new) goes back
        # into review as disappearing / newly appearing instead of silently dropping out of the state.
        for lid in bl:
            if not any(c["baseline_lesion_id"] == lid for c in corr):
                corr.append({"baseline_lesion_id": lid, "followup_lesion_id": None, "status": "disappearing", "confidence": None,
                             "centroid_distance_mm": None, "change_class": None, "source": src, "event_id": event["event_id"]})
        for lid in fu:
            if not any(c["followup_lesion_id"] == lid for c in corr):
                corr.append({"baseline_lesion_id": None, "followup_lesion_id": lid, "status": "newly_appearing", "confidence": None,
                             "centroid_distance_mm": None, "change_class": None, "source": src, "event_id": event["event_id"]})
    elif et == "disappeared_lesion" and a is not None:
        old_b = event.get("original_followup_lesion_id")
        drop(lambda c: c["baseline_lesion_id"] == a)
        corr.append({"baseline_lesion_id": a, "followup_lesion_id": None, "status": "disappearing", "confidence": None,
                     "centroid_distance_mm": None, "change_class": None, "source": src, "event_id": event["event_id"],
                     "verified_by_event": event["event_id"]})
        if old_b is not None and not any(c["followup_lesion_id"] == old_b for c in corr):
            corr.append({"baseline_lesion_id": None, "followup_lesion_id": old_b, "status": "newly_appearing", "confidence": None,
                         "centroid_distance_mm": None, "change_class": None, "source": src, "event_id": event["event_id"]})
    elif et == "new_lesion":
        b = event.get("original_followup_lesion_id") or event.get("corrected_followup_lesion_id")
        old_a = a
        drop(lambda c: c["followup_lesion_id"] == b)
        corr.append({"baseline_lesion_id": None, "followup_lesion_id": b, "status": "newly_appearing", "confidence": None,
                     "centroid_distance_mm": None, "change_class": None, "source": src, "event_id": event["event_id"],
                     "verified_by_event": event["event_id"]})
        if old_a is not None and not any(c["baseline_lesion_id"] == old_a for c in corr):
            corr.append({"baseline_lesion_id": old_a, "followup_lesion_id": None, "status": "disappearing", "confidence": None,
                         "centroid_distance_mm": None, "change_class": None, "source": src, "event_id": event["event_id"]})
    elif et == "measurement" and event.get("corrected_value") is not None:
        side = event.get("measurement_side", "followup")
        lid = a if side == "baseline" else event.get("original_followup_lesion_id")
        les = bl if side == "baseline" else fu
        if lid in les:
            les[lid]["diameter_mm_corrected"] = float(event["corrected_value"])
    elif et == "lesion_match" and action == "accept":
        for c in corr:
            if c["baseline_lesion_id"] == a and c["followup_lesion_id"] == event.get("original_followup_lesion_id"):
                c["verified_by_event"] = event["event_id"]
    # uncertain / segmentation_quality / lesion_status: recorded in the history only.

    new["correction_history"].append({"event_id": event["event_id"], "event_type": et, "action": action,
                                      "correction_source": src, "timestamp": event["timestamp"]})
    new["provenance"] = {**new["provenance"], "updated": now_iso(), "updated_by": f"{src}:{event.get('clinician_id')}",
                         "parent_state_sha256": sha256_json(state)}
    return new
