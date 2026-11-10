"""Clinician context supporting neutral, reviewer-specific cohort ordering."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import yaml

from adapters.schema_mapping import load_bundle
from configs.loader import resolve
from corrections.event_store import EventStore
from repair.matching_repair import CentroidHungarianMatcher, matcher_from_dict
from state.state_versioning import StateStore

from .ui_utils import ReviewContext, assert_no_gt

REVIEWER_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")


class CohortReviewContext(ReviewContext):
    """Review context with configurable real-cohort size and reviewer shuffling."""

    @classmethod
    def from_config(cls, ui_config: str | Path = "configs/ui.yaml") -> "CohortReviewContext":
        cfg = yaml.safe_load(resolve(ui_config).read_text(encoding="utf-8"))
        root = resolve(cfg["bundle"])
        bundle = load_bundle(root)
        events = EventStore(resolve(cfg["events_dir"]))
        states = StateStore(resolve(cfg["states_dir"]))
        sessions = resolve(cfg.get("sessions_dir", "exports/ui_sessions"))
        sessions.mkdir(parents=True, exist_ok=True)
        model_version = cfg.get("model_version", "V0")
        registry = resolve(cfg.get("versions_dir", "exports/versions")) / model_version / "model.json"
        matcher = (matcher_from_dict(json.loads(registry.read_text(encoding="utf-8"))) if registry.is_file()
                   else CentroidHungarianMatcher(30.0))
        configured = cfg.get("review_patients") or [p["patient_id"] for p in bundle["manifest"]["patients"]]
        patients = [patient_id for patient_id in configured if patient_id in bundle["cases"]]
        context = cls(
            cfg, root, bundle["cases"], cfg.get("data_mode", bundle["manifest"]["data_mode"]),
            events, states, matcher, model_version, resolve(cfg["events_dir"]) / "case_status.json",
            sessions, patients,
        )
        if context.data_mode == "real":
            expected = int(cfg.get("expected_patient_count", bundle["manifest"].get("n_patients", len(patients))))
            if bundle["manifest"].get("n_patients") != expected or len(patients) != expected:
                raise ValueError(f"real clinician configuration must contain exactly {expected} patients")
            if any(patient_id.upper().startswith("SYN") for patient_id in patients):
                raise ValueError("synthetic patient found in real clinician configuration")
            for case in context.cases.values():
                assert_no_gt(case, f"case:{case.get('patient_id')}")
        return context

    def patient_order(self, clinician_id: str) -> list[str]:
        """Return a deterministic independent order for each reviewer pseudonym."""
        patients = list(self.patients)
        if not self.cfg.get("randomize_case_order") or not clinician_id:
            return patients
        namespace = str(self.cfg.get("order_seed_namespace", "exp26-review-order-v1"))
        digest = hashlib.sha256(f"{namespace}\0{clinician_id}".encode("utf-8")).digest()
        random.Random(int.from_bytes(digest[:16], "big")).shuffle(patients)
        return patients

    def use_reviewer(self, clinician_id: str) -> None:
        """Give each reviewer pseudonym its own event log, state versions and case statuses.

        Reviewers sharing one server must never see each other's decisions or corrected states; otherwise the second
        reviewer would inherit already-decided items and a state the first reviewer has already changed.
        """
        if not clinician_id or not self.cfg.get("per_reviewer_stores"):
            return
        if not REVIEWER_ID.fullmatch(clinician_id):
            raise ValueError("Clinician ID may contain only letters, digits, '-' and '_' (max 40 characters).")
        events_root = resolve(self.cfg["events_dir"]) / clinician_id
        self.events = EventStore(events_root)
        self.states = StateStore(resolve(self.cfg["states_dir"]) / clinician_id)
        self.status_path = events_root / "case_status.json"
