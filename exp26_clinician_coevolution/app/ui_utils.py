"""Shared clinician-UI helpers.

This module and the entire ``app`` package only load the configured clinician
bundle. They must never import an evaluation-annotation loader or traverse any
server-side evaluation directory.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st
import yaml
from PIL import Image, ImageDraw

from adapters.schema_mapping import load_bundle
from configs.loader import resolve
from corrections.event_store import EventStore
from repair.matching_repair import CentroidHungarianMatcher, matcher_from_dict
from state.patient_state import build_state
from state.schemas import GT_FIELD_NAMES
from state.state_versioning import StateStore

BANNER = "Research use only — not for clinical decision-making."
CSS = """
<style>
.stApp {background:#ffffff; color:#1a1a19;}
.research-banner {background:#E8EDF7; color:#1f3770; border-left:4px solid #3B5BA5; padding:6px 12px; font-size:13px;
                  font-weight:600; margin-bottom:10px; border-radius:2px;}
.small-muted {color:#5f5e5a; font-size:12px;}
.metric-grid td {padding:2px 10px 2px 0; font-size:13px;}
h1, h2, h3 {color:#1a1a19;}
div.stButton > button {border-radius:3px;}
</style>
"""

UI_LESION_FIELDS = ("lesion_id", "organ", "centroid_mm", "centroid_registered_mm", "volume_mm3", "diameter_mm", "crop_ref", "mask_ref",
                    "segmentation_confidence", "diameter_mm_corrected")
UI_STATE_FIELDS = ("patient_id", "state_version", "source_model_version", "data_mode", "baseline_exam", "followup_exam",
                   "baseline_lesions", "followup_lesions", "correspondences", "patient_level", "correction_history")


def assert_no_gt(obj, path="payload") -> None:
    if isinstance(obj, dict):
        bad = GT_FIELD_NAMES & set(obj)
        if bad:
            raise AssertionError(f"GT field(s) {sorted(bad)} in clinician-facing {path}")
        for key, value in obj.items():
            assert_no_gt(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            assert_no_gt(value, f"{path}[{i}]")


def ui_payload(state: dict) -> dict:
    """Whitelisted, annotation-free view of a patient state for the UI."""
    out = {key: state[key] for key in UI_STATE_FIELDS if key in state}
    out["baseline_lesions"] = [{key: lesion.get(key) for key in UI_LESION_FIELDS if key in lesion}
                               for lesion in state["baseline_lesions"]]
    out["followup_lesions"] = [{key: lesion.get(key) for key in UI_LESION_FIELDS if key in lesion}
                               for lesion in state["followup_lesions"]]
    assert_no_gt(out)
    return out


def _frozen_prediction(case: dict) -> dict | None:
    """Decode the prediction serialized at pilot-build time, if present."""
    frozen = case.get("v0_prediction")
    if not frozen:
        return None
    pairs = [tuple(pair) for pair in frozen["pairs"]]
    confidence = {tuple(key.split("|", 1)): float(value) for key, value in frozen.get("confidence", {}).items()}
    distance = {tuple(key.split("|", 1)): float(value) for key, value in frozen.get("centroid_distance_mm", {}).items()}
    return {"pairs": pairs, "confidence": confidence, "distance": distance}


@dataclass
class ReviewContext:
    cfg: dict
    root: Path
    cases: dict
    data_mode: str
    events: EventStore
    states: StateStore
    matcher: object
    model_version: str
    status_path: Path
    sessions_dir: Path
    patients: list = field(default_factory=list)

    @classmethod
    def from_config(cls, ui_config: str | Path = "configs/ui.yaml") -> "ReviewContext":
        cfg = yaml.safe_load(resolve(ui_config).read_text(encoding="utf-8"))
        root = resolve(cfg["bundle"])
        bundle = load_bundle(root)
        events = EventStore(resolve(cfg["events_dir"]))
        states = StateStore(resolve(cfg["states_dir"]))
        sessions = resolve(cfg.get("sessions_dir", "exports/ui_sessions"))
        sessions.mkdir(parents=True, exist_ok=True)
        mv = cfg.get("model_version", "V0")
        reg = resolve(cfg.get("versions_dir", "exports/versions")) / mv / "model.json"
        matcher = matcher_from_dict(json.loads(reg.read_text(encoding="utf-8"))) if reg.is_file() else CentroidHungarianMatcher(30.0)
        pts = cfg.get("review_patients") or [p["patient_id"] for p in bundle["manifest"]["patients"]]
        context = cls(cfg, root, bundle["cases"], cfg.get("data_mode", bundle["manifest"]["data_mode"]), events, states,
                      matcher, mv, resolve(cfg["events_dir"]) / "case_status.json", sessions,
                      [p for p in pts if p in bundle["cases"]])
        if context.data_mode == "real":
            if bundle["manifest"].get("n_patients") != 10 or len(context.patients) != 10:
                raise ValueError("real pilot configuration must contain exactly 10 patients")
            if any(pid.upper().startswith("SYN") for pid in context.patients):
                raise ValueError("synthetic patient found in real pilot configuration")
            for case in context.cases.values():
                assert_no_gt(case, f"case:{case.get('patient_id')}")
        return context

    def current_state(self, pid: str) -> dict:
        if not self.states.versions(pid):
            pred = _frozen_prediction(self.cases[pid]) or self.matcher.predict(self.cases[pid])
            state = build_state(self.cases[pid], pred["pairs"], self.model_version, pred.get("confidence"), pred.get("distance"),
                                self.cases[pid].get("tool_versions"))
            self.states.init_patient(state)
        return self.states.load(pid)

    def original_state(self, pid: str) -> dict:
        self.current_state(pid)
        return self.states.load(pid, 0)

    def _status_file(self) -> dict:
        return json.loads(self.status_path.read_text(encoding="utf-8")) if self.status_path.exists() else {}

    def set_status(self, pid: str, status: str, clinician_id: str) -> None:
        statuses = self._status_file()
        statuses[pid] = {"status": status, "clinician_id": clinician_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(statuses, indent=1, sort_keys=True), encoding="utf-8")

    def status(self, pid: str) -> str:
        explicit = self._status_file().get(pid, {}).get("status")
        if explicit:
            return explicit
        return "In progress" if self.events.for_patient(pid) else "Not reviewed"

    def batch_of(self, pid: str) -> str:
        return self.cfg.get("review_batch", "demo_review")

    def touch_session(self, session_id: str, clinician_id: str, pid: str, page: str) -> None:
        path = self.sessions_dir / f"{session_id}.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "session_id": session_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "data_mode": self.data_mode,
        }
        record.update({"clinician_id": clinician_id or None, "last_patient_id": pid,
                       "last_page": page, "last_seen": time.strftime("%Y-%m-%dT%H:%M:%S")})
        path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")


def load_crop(ctx: ReviewContext, rel: str | None):
    if not rel:
        return None
    path = ctx.root / rel
    return Image.open(path) if path.is_file() else None


@st.cache_data(show_spinner=False, max_entries=24)
def _load_npz(path: str, modified_ns: int) -> tuple:
    del modified_ns
    with np.load(path, allow_pickle=False) as archive:
        image = archive["image_hu"] if "image_hu" in archive else archive["image"]
        label = archive["label"]
        ids = tuple(str(x) for x in archive["lesion_ids"])
        centroids = archive["centroid_zyx"] if "centroid_zyx" in archive else np.empty((0, 3), np.float32)
        spacing = archive["spacing_zyx"] if "spacing_zyx" in archive else np.ones(3, np.float32)
    return image, label, ids, centroids, spacing


def volume(ctx: ReviewContext, exam: dict):
    ref = exam.get("image_ref")
    if not ref or not str(ref).endswith(".npz"):
        return None
    path = ctx.root / ref
    if not path.is_file():
        return None
    return _load_npz(str(path), path.stat().st_mtime_ns)


def render_slice(img: np.ndarray, lab: np.ndarray, ids: tuple[str, ...], centroids: np.ndarray, spacing: np.ndarray,
                 k: int, show_mask: bool, show_centroids: bool, show_ids: bool,
                 window_center: float = 40.0, window_width: float = 400.0,
                 highlight: set | None = None, size: int = 512) -> Image.Image:
    del spacing
    source = img[k].astype(np.float32)
    if img.dtype == np.uint8 and float(img.min()) >= 0 and float(img.max()) <= 255:
        gray = source
    else:
        lo = window_center - max(window_width, 1.0) / 2.0
        gray = np.clip((source - lo) * (255.0 / max(window_width, 1.0)), 0, 255)
    rgb = np.stack([gray] * 3, axis=-1)
    if show_mask:
        mask = lab[k] > 0
        rgb[mask] = rgb[mask] * 0.55 + np.array([59, 91, 165]) * 0.45
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    sy, sx = size / source.shape[0], size / source.shape[1]
    for index, lesion_id in enumerate(ids, start=1):
        if index - 1 < len(centroids):
            z, y, x = centroids[index - 1]
        else:
            points = np.argwhere(lab == index)
            if not len(points):
                continue
            z, y, x = points.mean(axis=0)
        if abs(float(z) - k) > 0.5:
            continue
        color = (230, 75, 53) if highlight and lesion_id in highlight else (255, 224, 90)
        cy, cx = float(y) * sy, float(x) * sx
        if show_centroids:
            draw.line((cx - 7, cy, cx + 7, cy), fill=color, width=2)
            draw.line((cx, cy - 7, cx, cy + 7), fill=color, width=2)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=color, width=2)
        if show_ids:
            draw.rectangle((cx + 5, cy - 10, cx + 12 + 7 * len(lesion_id), cy + 8), fill=(0, 0, 0))
            draw.text((cx + 8, cy - 8), lesion_id, fill=color)
    return image
