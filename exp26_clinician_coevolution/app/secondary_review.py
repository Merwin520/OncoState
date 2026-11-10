"""Page D — optional segmentation, diameter, and lesion-status review."""
from __future__ import annotations

import time

import streamlit as st

from corrections.correction_event import clinician_event, mint_ui_action_token
from state.state_versioning import apply_event
from .ui_utils import load_crop, ui_payload


def _elapsed(key: str) -> float:
    starts = st.session_state.setdefault("secondary_start", {})
    started = starts.setdefault(key, time.time())
    return round(time.time() - started, 2)


def _record(ctx, pid, clinician_id, action, lesion_id, side, corrected_value=None, comment=None, review_seconds=None):
    state = ctx.current_state(pid)
    event = clinician_event(
        clinician_id=clinician_id, ui_action_token=mint_ui_action_token(), session_id=st.session_state.get("session_id"),
        patient_id=pid, batch_id=ctx.batch_of(pid), action=action, model_version=state["source_model_version"],
        state_version_before=state["state_version"], review_item=f"secondary:{action}:{lesion_id}:{st.session_state.get('sec_n', 0)}",
        baseline_lesion_id=lesion_id if side == "baseline" else None,
        original_followup_lesion_id=lesion_id if side == "followup" else None,
        original_prediction={"lesion_id": lesion_id, "side": side}, corrected_value=corrected_value,
        measurement_side=side if action == "measurement" else None, clinician_comment=comment,
        review_time_seconds=review_seconds, data_mode=ctx.data_mode,
    )
    sealed = ctx.events.append(event)
    ctx.states.write(apply_event(state, sealed))
    st.session_state["sec_n"] = st.session_state.get("sec_n", 0) + 1


def render(ctx, pid: str, clinician_id: str) -> None:
    st.subheader(f"D · Secondary review (optional) — {pid}")
    if not clinician_id:
        st.warning("Enter your study clinician ID in the sidebar.")
        return
    state = ui_payload(ctx.current_state(pid))
    side = st.radio("Exam", ["baseline", "followup"], horizontal=True, key="d_side")
    lesions = state["baseline_lesions" if side == "baseline" else "followup_lesions"]
    if not lesions:
        st.info("No lesions on this exam.")
        return
    lesion_id = st.selectbox("Lesion", [lesion["lesion_id"] for lesion in lesions], key="d_lesion")
    lesion = next(item for item in lesions if item["lesion_id"] == lesion_id)
    timer_key = f"{pid}:{side}:{lesion_id}"
    _elapsed(timer_key)
    image = load_crop(ctx, lesion.get("crop_ref"))
    if image is not None:
        st.image(image, width=240)
    c1, c2 = st.columns(2)
    if c1.button("Segmentation acceptable", key=f"seg_ok_{side}_{lesion_id}"):
        _record(ctx, pid, clinician_id, "segmentation_ok", lesion_id, side, review_seconds=_elapsed(timer_key))
        st.success("Recorded.")
    if c2.button("Segmentation incorrect", key=f"seg_bad_{side}_{lesion_id}"):
        _record(ctx, pid, clinician_id, "segmentation_incorrect", lesion_id, side, review_seconds=_elapsed(timer_key))
        st.success("Recorded.")
    st.markdown(f"Predicted diameter: **{lesion['diameter_mm']:.1f} mm**")
    new_diameter = st.number_input("Corrected diameter (mm)", min_value=0.0, value=float(round(lesion["diameter_mm"], 1)),
                                   step=0.5, key=f"d_{side}_{lesion_id}")
    if st.button("Save corrected diameter", key=f"save_d_{side}_{lesion_id}"):
        _record(ctx, pid, clinician_id, "measurement", lesion_id, side, corrected_value=float(new_diameter),
                review_seconds=_elapsed(timer_key))
        st.success("Recorded.")
    current = next((item["status"] for item in state["correspondences"]
                    if (item["baseline_lesion_id"] if side == "baseline" else item["followup_lesion_id"]) == lesion_id), "—")
    st.markdown(f"Predicted lesion status: **{current}**")
    status = st.selectbox("Corrected status", ["persistent", "disappearing", "newly_appearing"], key=f"st_{side}_{lesion_id}")
    if st.button("Save corrected status", key=f"save_st_{side}_{lesion_id}"):
        _record(ctx, pid, clinician_id, "status", lesion_id, side, corrected_value=status, review_seconds=_elapsed(timer_key))
        st.success("Recorded (matching changes are made on page C).")
