"""Page C — matching review (primary interaction): "Are these the same lesion?"

Every decision mints a UI action token and creates exactly one immutable clinician event plus a new patient-state
version. The model's original prediction (state v0) is never modified, and no model parameter changes here.

Correction candidates are ALL lesions of the other exam, ranked by registered centroid distance. There is deliberately
no distance cutoff: when registration fails (residual of 100+ mm) the true partner can be far away in registered space,
and the clinician must still be able to select it.
"""
from __future__ import annotations

import math
import time

import streamlit as st

from corrections.correction_event import clinician_event, mint_ui_action_token
from corrections.validation import validate_against_state
from state.review_items import alternative_candidates, nearest_baselines, review_items
from state.state_versioning import apply_event
from .ui_utils import load_crop, render_slice, ui_payload, volume

ALL = 10 ** 6
POOR_REGISTRATION_MM = 30.0
UNIDENTIFIED_TAG = "[rejected match; correct lesion not identifiable]"
WINDOWS = {"Soft tissue": (40.0, 400.0), "Lung": (-600.0, 1500.0), "Liver": (60.0, 160.0), "Bone": (400.0, 1800.0)}


def open_items(ctx, pid: str) -> list[dict]:
    reviewed = {e["review_item"] for e in ctx.events.for_patient(pid)}
    # A correspondence created or confirmed by a clinician decision is already decided; lesions released by a re-pairing
    # come back unverified and are asked about again.
    return [i for i in review_items(ctx.current_state(pid)) if i["item_id"] not in reviewed and not i["verified"]]


def _partner(state: dict, baseline_lesion_id: str) -> str | None:
    return next((c["followup_lesion_id"] for c in state["correspondences"] if c["baseline_lesion_id"] == baseline_lesion_id), None)


def record_decision(ctx, pid: str, item: dict, action: str, clinician_id: str, ui_token: str, corrected_fu: str | None = None,
                    corrected_bl: str | None = None, comment: str | None = None, review_seconds: float | None = None) -> dict:
    """Create the clinician event, append it, and write the new patient-state version. Returns the sealed event.

    For a "new" item corrected to "continues baseline lesion X", the event is expressed from X's point of view:
    X's current partner (usually none) is the original follow-up lesion and the new lesion is the corrected one.
    """
    state = ctx.current_state(pid)
    a = corrected_bl if corrected_bl is not None else item["baseline_lesion_id"]
    original_fu = _partner(state, corrected_bl) if corrected_bl is not None else item["followup_lesion_id"]
    ev = clinician_event(
        clinician_id=clinician_id, ui_action_token=ui_token, session_id=st.session_state.get("session_id") if _has_st() else None,
        patient_id=pid, batch_id=ctx.batch_of(pid), action=action, model_version=state["source_model_version"],
        state_version_before=state["state_version"], review_item=item["item_id"], baseline_lesion_id=a,
        original_followup_lesion_id=original_fu, corrected_followup_lesion_id=corrected_fu,
        original_prediction={"kind": item["kind"], "baseline_lesion_id": item["baseline_lesion_id"], "followup_lesion_id": item["followup_lesion_id"]},
        corrected_value={"followup_lesion_id": corrected_fu, "baseline_lesion_id": a} if action == "correct" else None,
        original_confidence=item.get("confidence"), registration_quality=state["patient_level"].get("registration_quality"),
        centroid_distance=item.get("centroid_distance_mm"), review_time_seconds=review_seconds, clinician_comment=comment,
        data_mode=ctx.data_mode)
    validate_against_state(ev, state)
    sealed = ctx.events.append(ev)
    ctx.states.write(apply_event(state, sealed))
    return sealed


def _has_st() -> bool:
    try:
        return st.runtime.exists()
    except Exception:  # noqa: BLE001
        return False


def _elapsed(key: str) -> float:
    starts = st.session_state.setdefault("item_start", {})
    t0 = starts.setdefault(key, time.time())
    return round(time.time() - t0, 2)


def _fmt(value, unit: str = "", digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}{unit}" if math.isfinite(number) else "—"


def candidates(state: dict, item: dict) -> list[dict]:
    """Every lesion on the other exam the clinician could pick instead, nearest (registered) first."""
    fu_partner = {c["followup_lesion_id"]: c for c in state["correspondences"] if c["followup_lesion_id"]}
    bl_partner = {c["baseline_lesion_id"]: c for c in state["correspondences"] if c["baseline_lesion_id"]}
    if item["kind"] == "new":
        out = []
        for x in nearest_baselines(state, item["followup_lesion_id"], k=ALL, max_mm=math.inf):
            c = bl_partner.get(x["baseline_lesion_id"])
            note = f"currently matched to {c['followup_lesion_id']}" if c and c["followup_lesion_id"] else "currently predicted disappeared"
            out.append({"lesion_id": x["baseline_lesion_id"], "side": "baseline", "distance_mm": x["distance_mm"], "note": note})
        return out
    out = []
    for x in alternative_candidates(state, item["baseline_lesion_id"], k=ALL, max_mm=math.inf):
        if x["followup_lesion_id"] == item["followup_lesion_id"]:
            continue
        c = fu_partner.get(x["followup_lesion_id"])
        note = f"currently matched to {c['baseline_lesion_id']}" if c and c["baseline_lesion_id"] else "currently predicted new"
        out.append({"lesion_id": x["followup_lesion_id"], "side": "followup", "distance_mm": x["distance_mm"], "note": note})
    return out


def _lesion_slice(vol, lesion_id: str | None) -> int | None:
    if vol is None or lesion_id is None or lesion_id not in vol[2]:
        return None
    return int(round(float(vol[3][vol[2].index(lesion_id)][0])))


def _full_ct(ctx, pid: str, s: dict, item: dict, cands: list[dict], key: str, selected: str | None) -> None:
    baseline, followup = volume(ctx, s["baseline_exam"]), volume(ctx, s["followup_exam"])
    if baseline is None or followup is None:
        st.error("A full CT volume is unavailable for this case.")
        return
    window = st.selectbox("CT window", list(WINDOWS), key="c_window")
    center, width = WINDOWS[window]
    a, b = item["baseline_lesion_id"], item["followup_lesion_id"]
    top = cands[0]["lesion_id"] if cands else None
    bl_focus = a or (selected or top if item["kind"] == "new" else None)
    fu_focus = b or (selected or top if item["kind"] != "new" else None)
    if item["kind"] == "pair" and selected:
        fu_focus = selected
    left, right = st.columns(2)
    for col, vol, focus, title, side in ((left, baseline, bl_focus, "Baseline", "bl"), (right, followup, fu_focus, "Follow-up", "fu")):
        n = vol[0].shape[0]
        z = _lesion_slice(vol, focus)
        z = n // 2 if z is None else min(max(z, 0), n - 1)
        k = col.slider(f"{title} axial slice", 0, n - 1, z, key=f"ct_{side}_{key}_{focus}")
        highlight = {x for x in (a, b, selected) if x}
        col.image(render_slice(*vol, k, True, True, True, center, width, highlight=highlight),
                  caption=f"{title} · axial {k + 1}/{n} · centred on {focus or '—'} · red = lesions in this question", width="stretch")


def _next_patient(ctx, pid: str) -> str | None:
    order = list(ctx.patients)
    rest = order[order.index(pid) + 1:] + order[:order.index(pid)] if pid in order else order
    return next((p for p in rest if ctx.status(p) not in ("Completed", "Excluded")), None)


def _complete_and_advance(ctx, pid: str, clinician_id: str, mark: bool) -> None:
    if mark:
        ctx.set_status(pid, "Completed", clinician_id)
    nxt = _next_patient(ctx, pid)
    if nxt:
        st.session_state["patient"] = nxt


def render(ctx, pid: str, clinician_id: str) -> None:
    st.subheader(f"C · Matching review — {pid}")
    if not clinician_id:
        st.warning("Enter your study clinician ID in the sidebar before reviewing.")
        return
    items = open_items(ctx, pid)
    decided = len({e["review_item"] for e in ctx.events.for_patient(pid)})
    total = decided + len(items)
    if not items:
        st.success(f"All {total} matching decisions for this patient are recorded.")
        done = ctx.status(pid) == "Completed"
        c1, c2 = st.columns(2)
        c1.button("Mark case completed and open next patient", type="primary", disabled=done, key=f"advance_{pid}",
                  on_click=_complete_and_advance, args=(ctx, pid, clinician_id, True))
        c2.button("Open next patient", key=f"next_{pid}", on_click=_complete_and_advance, args=(ctx, pid, clinician_id, False))
        st.caption("Page D (segmentation / diameter) is optional. Page E lets you flag or exclude a case.")
        return
    item = items[0]
    key = f"{pid}|{item['item_id']}"
    _elapsed(key)
    state = ctx.current_state(pid)
    s = ui_payload(state)
    bl = {l["lesion_id"]: l for l in s["baseline_lesions"]}
    fu = {l["lesion_id"]: l for l in s["followup_lesions"]}
    st.progress(decided / total if total else 0.0, text=f"Decision {decided + 1} of {total} for this patient")
    a, b = item["baseline_lesion_id"], item["followup_lesion_id"]
    q = {"pair": "Are these the same lesion?",
         "disappeared": "The model predicts this baseline lesion has DISAPPEARED at follow-up. Is that correct?",
         "new": "The model predicts this follow-up lesion is NEW (not present at baseline). Is that correct?"}[item["kind"]]
    st.markdown(f"#### {q}")
    rq = s["patient_level"].get("registration_quality", {}).get("residual_mm")
    try:
        poor = rq is not None and float(rq) > POOR_REGISTRATION_MM
    except (TypeError, ValueError):
        poor = False
    if poor:
        st.warning(f"Automatic registration between the two exams is poor for this patient (residual {_fmt(rq, ' mm')}). "
                   "Distances and candidate ranking may be misleading — please judge by anatomy, using the full CT below.")
    left, right = st.columns(2)
    for col, lid, les, title in ((left, a, bl, "Baseline"), (right, b, fu, "Follow-up")):
        col.markdown(f"**{title}: {lid or '— (no lesion predicted)'}**")
        if lid:
            im = load_crop(ctx, les[lid].get("crop_ref"))
            if im is not None:
                col.image(im, width=260)
            col.markdown(f"<span class='small-muted'>diameter {_fmt(les[lid]['diameter_mm'], ' mm')} · "
                         f"volume {_fmt(les[lid]['volume_mm3'] / 1000, ' mL', 2)}</span>", unsafe_allow_html=True)
    conf = item.get("confidence")
    st.markdown(f"<table class='metric-grid'><tr><td>centroid distance</td><td><b>{_fmt(item.get('centroid_distance_mm'), ' mm')}</b></td>"
                f"<td>matching confidence</td><td><b>{_fmt(conf, '', 2)}</b></td>"
                f"<td>registration residual</td><td><b>{_fmt(rq, ' mm')}</b></td>"
                f"<td>predicted status</td><td><b>{item['kind']}</b></td></tr></table>", unsafe_allow_html=True)
    comment = st.text_input("Comment (optional)", key=f"c_{key}")
    if item["kind"] == "pair":
        acts = [("Accept match", "accept"), ("Incorrect match", "incorrect"), ("Disappeared", "disappeared"),
                ("New lesion", "new_lesion"), ("Uncertain", "uncertain")]
    elif item["kind"] == "disappeared":
        acts = [("Confirm disappeared", "disappeared"), ("Matches a follow-up lesion", "incorrect"), ("Uncertain", "uncertain")]
    else:
        acts = [("Confirm new lesion", "new_lesion"), ("Continues a baseline lesion", "incorrect_new"), ("Uncertain", "uncertain")]
    cols = st.columns(5)
    for col, (label, act) in zip(cols, acts):
        if col.button(label, key=f"btn_{act}_{key}", width="stretch"):
            st.session_state["pending"] = {"item": key, "act": act}
    cands = candidates(state, item)
    selected = None
    pend = st.session_state.get("pending")
    if pend and pend["item"] == key:
        act = pend["act"]
        if act in ("incorrect", "incorrect_new"):
            side_title = "baseline lesion it continues" if act == "incorrect_new" else "correct follow-up lesion"
            st.markdown(f"**Select the {side_title}** (all lesions of the other exam, nearest first):")
            if not cands:
                st.info("The other exam has no other predicted lesion. Use the remaining buttons below.")
            else:
                lesions = bl if act == "incorrect_new" else fu
                per_row = 6
                for start in range(0, len(cands), per_row):
                    row = st.columns(per_row)
                    for colx, x in zip(row, cands[start:start + per_row]):
                        im = load_crop(ctx, lesions[x["lesion_id"]].get("crop_ref"))
                        if im is not None:
                            colx.image(im, width=140)
                        colx.caption(f"**{x['lesion_id']}** · {_fmt(x['distance_mm'], ' mm')} · "
                                     f"⌀ {_fmt(lesions[x['lesion_id']]['diameter_mm'], ' mm')}")
                labels = {f"{x['lesion_id']} · {_fmt(x['distance_mm'], ' mm')} away · ⌀ {_fmt(lesions[x['lesion_id']]['diameter_mm'], ' mm')}"
                          f" · {x['note']}": x["lesion_id"] for x in cands}
                choice = st.radio("Candidate", list(labels), key=f"pick_{key}")
                selected = labels[choice]
            b1, b2, b3 = st.columns(3)
            if cands and b1.button("Confirm correction", key=f"confirm_{key}", type="primary", width="stretch"):
                tok = mint_ui_action_token()
                if act == "incorrect":
                    record_decision(ctx, pid, item, "correct", clinician_id, tok, corrected_fu=selected, comment=comment or None,
                                    review_seconds=_elapsed(key))
                else:
                    record_decision(ctx, pid, item, "correct", clinician_id, tok, corrected_fu=b, corrected_bl=selected,
                                    comment=comment or None, review_seconds=_elapsed(key))
                st.session_state.pop("pending", None)
                st.rerun()
            if item["kind"] == "pair":
                if b2.button("None of these — baseline lesion has disappeared", key=f"none_{key}", width="stretch"):
                    record_decision(ctx, pid, item, "disappeared", clinician_id, mint_ui_action_token(), comment=comment or None,
                                    review_seconds=_elapsed(key))
                    st.session_state.pop("pending", None)
                    st.rerun()
                if b3.button("Wrong match, but I cannot identify the correct lesion", key=f"unid_{key}", width="stretch"):
                    note = f"{UNIDENTIFIED_TAG} {comment}".strip()
                    record_decision(ctx, pid, item, "uncertain", clinician_id, mint_ui_action_token(), comment=note,
                                    review_seconds=_elapsed(key))
                    st.session_state.pop("pending", None)
                    st.rerun()
            if st.button("Cancel", key=f"cancel_{key}"):
                st.session_state.pop("pending", None)
                st.rerun()
        else:
            tok = mint_ui_action_token()
            action = {"accept": "accept", "disappeared": "disappeared", "new_lesion": "new_lesion", "uncertain": "uncertain"}[act]
            record_decision(ctx, pid, item, action, clinician_id, tok, comment=comment or None, review_seconds=_elapsed(key))
            st.session_state.pop("pending", None)
            st.rerun()
    st.divider()
    if st.toggle("Show full CT (scroll slices around these lesions)", key="c_fullct"):
        _full_ct(ctx, pid, s, item, cands, key, selected)
