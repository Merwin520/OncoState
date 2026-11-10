"""OncoState Experiment 2.6 — final clinician study interface: ONE PATIENT = ONE PAGE.

Start screen (reviewer ID) -> Case k of 50 (prior CT | follow-up CT, all V0 lesion correspondences, edit only what is
wrong, save) -> next case -> "Review complete". No page navigation, no model statistics, no reference data.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != APP_DIR]
sys.path.insert(0, str(ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import one_page as op  # noqa: E402
from app.cohort_context import REVIEWER_ID, CohortReviewContext  # noqa: E402
from app.review_export import write_reviewer_csv  # noqa: E402
from configs.loader import resolve  # noqa: E402
from app.ui_utils import BANNER, load_crop, render_slice, volume  # noqa: E402
from corrections.correction_event import clinician_event, mint_ui_action_token  # noqa: E402
from corrections.validation import validate_against_state  # noqa: E402
from state.state_versioning import apply_event  # noqa: E402

CONFIG = os.environ.get("EXP26_UI_CONFIG") or next(
    (c for c in ("configs/ui_study.yaml", "configs/ui_clinician50.yaml") if (ROOT / c).is_file()), "configs/ui_clinician50.yaml")
WINDOWS = {"Soft tissue (W400 / L40)": (40.0, 400.0), "Lung (W1500 / L-600)": (-600.0, 1500.0), "Manual": None}
IDLE_CAP_S = 600.0  # an action more than 10 min after the previous one counts as 10 min (breaks are not review time)
TASK = """
**TASK** — Compare the prior and follow-up CT examinations. The AI has proposed which lesions correspond across the two
examinations. If a proposed match is correct, leave it unchanged. If it is incorrect, select the correct follow-up lesion.
You may also mark a prior lesion as disappeared, identify an unmatched follow-up lesion as new, or mark a lesion as
uncertain. Review the full CT images before saving the patient.
"""
CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
.banner {background:#E8EDF7; color:#1f3770; border-left:4px solid #3B5BA5; padding:5px 12px; font-size:13px; font-weight:600;
         border-radius:2px; margin-bottom:6px;}
.muted {color:#5f5e5a; font-size:13px;}
.conflict {color:#a3271c; font-weight:600;}
div.stButton > button {border-radius:4px;}
</style>
"""


# ----------------------------------------------------------------------------------------------------------- state
def _ctx(reviewer: str | None) -> CohortReviewContext:
    ctx = CohortReviewContext.from_config(CONFIG)
    if reviewer:
        ctx.use_reviewer(reviewer)
        ctx.patients = ctx.patient_order(reviewer)
    return ctx


def _progress_path(ctx) -> Path:
    return ctx.status_path.parent / "patient_progress.json"


def _progress(ctx) -> dict:
    path = _progress_path(ctx)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_progress(ctx, data: dict) -> None:
    path = _progress_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")


def _record_order(ctx, reviewer: str) -> None:
    path = ctx.status_path.parent / "case_order.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"reviewer_id": reviewer, "order_seed_namespace": ctx.cfg.get("order_seed_namespace"),
                                    "case_order": ctx.patients}, indent=1), encoding="utf-8")


def _csv_path(ctx, reviewer: str) -> Path:
    return resolve(ctx.cfg["events_dir"]).parent / f"{reviewer}_review_results.csv"


def _refresh_csvs(ctx, reviewer: str) -> None:
    """Spreadsheet copies of this reviewer's decisions; never allowed to interrupt the review (e.g. file open in Excel)."""
    try:
        write_reviewer_csv(ctx.status_path.parent, ctx.cases, _csv_path(ctx, reviewer))
    except Exception:  # noqa: BLE001
        st.session_state["csv_stale"] = True
    else:
        st.session_state.pop("csv_stale", None)


def _current_case(ctx) -> str | None:
    return next((p for p in ctx.patients if ctx.status(p) != "Completed"), None)


def _seconds_since_last(pid: str) -> float:
    key = f"t_last_{pid}"
    now = time.time()
    elapsed = min(now - st.session_state.get(key, now), IDLE_CAP_S)
    st.session_state[key] = now
    return round(elapsed, 2)


def _write_event(ctx, reviewer: str, pid: str, rid: str, decision: str, target: str | None, comment: str | None = None,
                 implicit: bool = False) -> None:
    case = ctx.cases[pid]
    state = ctx.current_state(pid)
    spec = op.event_for(case, rid, decision, target)
    event = clinician_event(
        clinician_id=reviewer, ui_action_token=mint_ui_action_token(), session_id=st.session_state.get("session_id"),
        patient_id=pid, batch_id=ctx.batch_of(pid), model_version="V0", state_version_before=state["state_version"],
        review_item=rid + (op.IMPLICIT if implicit else ""), registration_quality=None, original_confidence=None,
        centroid_distance=None, review_time_seconds=0.0 if implicit else _seconds_since_last(pid),
        clinician_comment=comment, data_mode=ctx.data_mode, **spec)
    validate_against_state(event, state)
    sealed = ctx.events.append(event)
    ctx.states.write(apply_event(state, sealed))


# ------------------------------------------------------------------------------------------------------------- views
def start_screen() -> None:
    st.title("OncoState Longitudinal Lesion Review")
    st.markdown(f"<div class='banner'>{BANNER}</div>", unsafe_allow_html=True)
    reviewer = st.text_input("Reviewer ID", key="reviewer_input", max_chars=40, placeholder="e.g. R01").strip()
    if st.button("START REVIEW", type="primary", disabled=not reviewer):
        if not REVIEWER_ID.fullmatch(reviewer):
            st.error("Reviewer ID may contain only letters, digits, '-' and '_'.")
            return
        st.session_state["reviewer"] = reviewer
        st.rerun()
    st.caption("Use the same Reviewer ID every time — your progress is saved automatically and resumes where you stopped.")


def sidebar(ctx, reviewer: str) -> None:
    done = sum(ctx.status(p) == "Completed" for p in ctx.patients)
    with st.sidebar:
        st.markdown(f"**Reviewer:** {reviewer}")
        st.progress(done / len(ctx.patients), text=f"{done} / {len(ctx.patients)} patients completed")
        st.caption("Every change is saved immediately. To stop, simply close this browser tab and the black window. "
                   "To resume, start the program again and enter the same Reviewer ID.")
        if st.session_state.get("csv_stale"):
            st.caption("The results CSV could not be updated (is it open in Excel?). It will update after the next saved patient.")
        if st.button("Switch reviewer"):
            st.session_state.clear()
            st.rerun()


def _focus_defaults(pid: str, rows: list[str]) -> str:
    focus = st.session_state.get(f"focus_{pid}")
    if focus not in rows:
        focus = rows[0] if rows else None
        st.session_state[f"focus_{pid}"] = focus
    return focus


def _jump(pid: str, side: str, vol, lesion_id: str | None) -> None:
    if vol is None or lesion_id is None or lesion_id not in vol[2]:
        return
    z = int(round(float(vol[3][vol[2].index(lesion_id)][0])))
    st.session_state[f"z_{side}_{pid}"] = min(max(z, 0), vol[0].shape[0] - 1)


def _select(pid: str, table: str, rows: list[str], other: str, bl_vol, fu_vol, case_r: dict) -> None:
    selection = st.session_state.get(f"{table}_{pid}_{st.session_state.get(f'ver_{table}_{pid}', 0)}", {})
    picked = (selection.get("selection") or {}).get("rows") or []
    if not picked:
        return
    rid = rows[picked[0]]
    st.session_state[f"focus_{pid}"] = rid
    st.session_state[f"ver_{other}_{pid}"] = st.session_state.get(f"ver_{other}_{pid}", 0) + 1  # clear the other table
    kind, lesion = rid.split(":", 1)
    if kind == "bl":
        _jump(pid, "bl", bl_vol, lesion)
        _jump(pid, "fu", fu_vol, case_r["target"].get(lesion) or case_r["v0"].get(lesion))
    else:
        _jump(pid, "fu", fu_vol, lesion)
        _jump(pid, "bl", bl_vol, case_r["fu_link"].get(lesion))


def viewer(pid: str, bl_vol, fu_vol, highlight: set) -> None:
    c = st.columns([2, 1, 1, 1])
    window = c[0].selectbox("CT window", list(WINDOWS), key="window")
    contour = c[1].toggle("Lesion contour", value=True, key="show_contour")
    ids = c[2].toggle("Lesion ID", value=True, key="show_ids")
    cent = c[3].toggle("Lesion centre", value=False, key="show_centre")
    if WINDOWS[window] is None:
        w1, w2 = st.columns(2)
        level = w1.number_input("Window level (HU)", -1200.0, 2000.0, 40.0, 10.0, key="wl")
        width = w2.number_input("Window width (HU)", 1.0, 4000.0, 400.0, 10.0, key="ww")
    else:
        level, width = WINDOWS[window]
    left, right = st.columns(2)
    for col, vol, side, title in ((left, bl_vol, "bl", "PRIOR CT"), (right, fu_vol, "fu", "FOLLOW-UP CT")):
        col.markdown(f"**{title}**")
        if vol is None:
            col.error("CT volume unavailable.")
            continue
        n = vol[0].shape[0]
        key = f"z_{side}_{pid}"
        if key not in st.session_state:
            st.session_state[key] = int(round(float(vol[3][0][0]))) if len(vol[3]) else n // 2
        k = col.slider("Axial slice", 0, n - 1, key=key, label_visibility="collapsed")
        col.image(render_slice(*vol, k, contour, cent, ids, level, width, highlight=highlight),
                  caption=f"axial {k + 1} / {n} · red = selected lesions", width="stretch")


def _size(lesions: dict, lid: str | None) -> str:
    return f"{lesions[lid]['diameter_mm']:.1f} mm" if lid and lid in lesions else "—"


def tables(ctx, pid: str, case: dict, r: dict, decided: dict, bl_vol, fu_vol) -> tuple[list[str], list[str]]:
    bl = {l["lesion_id"]: l for l in case["baseline_lesions"]}
    fu = {l["lesion_id"]: l for l in case["followup_lesions"]}
    st.markdown("#### AI-proposed longitudinal matches")
    st.caption("Click a row to show both lesions in the CT viewers above and to review or change it.")
    bl_rows, rows = [], []
    for b in r["bl_ids"]:
        ai = r["v0"].get(b)
        d = decided.get(f"bl:{b}")
        tgt = r["target"][b]
        if tgt in r["conflicts"]:
            review = f"⚠ {tgt} is also assigned to {', '.join(x for x in r['conflicts'][tgt] if x != b)}"
        elif d is None or d["decision"] == "keep":
            review = "✓ unchanged"
        elif d["decision"] == "changed_match":
            review = f"✎ changed → {tgt}"
        else:
            review = {"disappeared": "✎ disappeared", "uncertain": "? uncertain"}[d["decision"]]
        rows.append({"Prior lesion": b, "AI follow-up lesion": ai or "— (disappeared)", "Prior size": _size(bl, b),
                     "Follow-up size": _size(fu, tgt or ai), "Clinician review": review})
        bl_rows.append(f"bl:{b}")
    ver = st.session_state.get(f"ver_bl_{pid}", 0)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", on_select=lambda: _select(pid, "bl", bl_rows, "fu", bl_vol, fu_vol, r),
                 selection_mode="single-row", key=f"bl_{pid}_{ver}")
    fu_rows, rows = [], []
    for f in r["unclaimed"]:
        d = decided.get(f"fu:{f}")
        origin = "new" if f not in r["v0"].values() else "unmatched after your change"
        if f in r["conflicts"]:
            review = f"⚠ {r['fu_link'][f]} already has a follow-up lesion"
        elif d is None or d["decision"] == "new":
            review = "✓ new lesion" if origin == "new" else "new lesion"
        elif d["decision"] == "match_baseline":
            review = f"✎ matches {d['target']}"
        else:
            review = "? uncertain"
        rows.append({"Follow-up lesion": f, "AI proposal": origin, "Follow-up size": _size(fu, f), "Clinician review": review})
        fu_rows.append(f"fu:{f}")
    if fu_rows:
        st.markdown("#### Unmatched follow-up lesions")
        ver = st.session_state.get(f"ver_fu_{pid}", 0)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", on_select=lambda: _select(pid, "fu", fu_rows, "bl", bl_vol, fu_vol, r),
                     selection_mode="single-row", key=f"fu_{pid}_{ver}")
    return bl_rows, fu_rows


def editor(ctx, reviewer: str, pid: str, case: dict, r: dict, decided: dict, focus: str | None, fu_vol, bl_vol) -> None:
    if focus is None:
        return
    bl = {l["lesion_id"]: l for l in case["baseline_lesions"]}
    fu = {l["lesion_id"]: l for l in case["followup_lesions"]}
    kind, lesion = focus.split(":", 1)
    current = decided.get(focus)
    st.markdown("---")
    if kind == "bl":
        ai = r["v0"].get(lesion)
        st.markdown(f"#### Prior lesion **{lesion}** — AI proposal: **{ai or 'disappeared'}**")
        options = [f"Keep AI proposal ({ai})" if ai else "Keep AI proposal (disappeared)", "Match another follow-up lesion"] + \
                  (["Disappeared"] if ai else []) + ["Uncertain"]
        code = {options[0]: "keep", "Match another follow-up lesion": "changed_match", "Disappeared": "disappeared", "Uncertain": "uncertain"}
        now = (current or {"decision": "keep"})["decision"]
        index = next((i for i, o in enumerate(options) if code[o] == now), 0)
        choice = st.radio("Correct outcome", options, index=index, key=f"radio_{pid}_{focus}", horizontal=True)
        target = None
        if code[choice] == "changed_match":
            others = [f for f in r["fu_ids"] if f != ai]
            owner = {f: b for b, f in r["target"].items() if f and b != lesion}
            label = {f: f"{f} · {fu[f]['diameter_mm']:.1f} mm" + (f"  (currently {owner[f]})" if f in owner else "") for f in others}
            if not others:
                st.info("The follow-up CT has no other lesion. Choose Disappeared or Uncertain.")
            else:
                start = others.index(current["target"]) if current and current.get("target") in others else 0
                target = st.selectbox("Correct follow-up lesion", others, index=start, format_func=label.get,
                                      key=f"pick_{pid}_{focus}", on_change=lambda: _jump(pid, "fu", fu_vol, st.session_state[f"pick_{pid}_{focus}"]))
        crops = [("Prior " + lesion, bl, lesion), (f"AI follow-up {ai}" if ai else "AI: disappeared", fu, ai)]
        if target:
            crops.append((f"Your choice {target}", fu, target))
        valid = code[choice] != "changed_match" or target is not None
    else:
        origin = "new lesion" if lesion not in r["v0"].values() else "unmatched after your change"
        st.markdown(f"#### Follow-up lesion **{lesion}** — AI proposal: **{origin}**")
        options = ["New lesion", "Matches a prior lesion", "Uncertain"]
        code = {"New lesion": "new", "Matches a prior lesion": "match_baseline", "Uncertain": "uncertain"}
        now = (current or {"decision": "new"})["decision"]
        index = next((i for i, o in enumerate(options) if code[o] == now), 0)
        choice = st.radio("Correct outcome", options, index=index, key=f"radio_{pid}_{focus}", horizontal=True)
        target = None
        if code[choice] == "match_baseline":
            free = [b for b in r["bl_ids"] if r["target"][b] is None]
            if not free:
                st.info("Every prior lesion already has a follow-up partner. Change that prior lesion's row first.")
            else:
                start = free.index(current["target"]) if current and current.get("target") in free else 0
                target = st.selectbox("Prior lesion it continues", free, index=start,
                                      format_func=lambda b: f"{b} · {bl[b]['diameter_mm']:.1f} mm", key=f"pick_{pid}_{focus}",
                                      on_change=lambda: _jump(pid, "bl", bl_vol, st.session_state[f"pick_{pid}_{focus}"]))
        crops = [(f"Follow-up {lesion}", fu, lesion)] + ([(f"Prior {target}", bl, target)] if target else [])
        valid = code[choice] != "match_baseline" or target is not None
    cols = st.columns(4)
    for col, (title, lesions, lid) in zip(cols, crops):
        col.caption(title)
        image = load_crop(ctx, lesions[lid].get("crop_ref")) if lid else None
        if image is not None:
            col.image(image, width=200)
    pending = (code[choice], target) != ((current or {}).get("decision", code[options[0]]), (current or {}).get("target"))
    if st.button("APPLY", type="primary", disabled=not (valid and pending), key=f"apply_{pid}_{focus}"):
        _write_event(ctx, reviewer, pid, focus, code[choice], target)
        st.rerun()


def save_block(ctx, reviewer: str, pid: str, case: dict, r: dict, decided: dict, index: int) -> None:
    st.markdown("---")
    n = op.counts(case, decided)
    st.markdown(f"AI matches unchanged: **{n['unchanged']}** · Clinician corrections: **{n['changed']}** · "
                f"Disappeared: **{n['disappeared']}** · New lesions: **{n['new']}** · Uncertain: **{n['uncertain']}**")
    if r["conflicts"]:
        st.markdown("<span class='conflict'>Some follow-up lesions are assigned twice (⚠ rows). Please resolve them before saving.</span>",
                    unsafe_allow_html=True)
    comment = st.text_input("Optional comment", key=f"comment_{pid}")
    if st.button("SAVE PATIENT & NEXT CASE", type="primary", disabled=bool(r["conflicts"]), key=f"save_{pid}"):
        for rid in op.implicit_rows(case, decided):
            kind = rid.split(":", 1)[0]
            _write_event(ctx, reviewer, pid, rid, "keep" if kind == "bl" else "new", None, implicit=True)
        final = op.decisions(ctx.events.for_patient(pid))
        progress = _progress(ctx)
        record = progress.get(pid, {})
        spent = record.get("seconds", 0.0) + _seconds_since_last(pid)
        record.update({"case_order": index + 1, "review_end": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(spent, 2),
                       "comment": comment or None, "final_pairs": op.resolve(case, final)["pairs"], "counts": op.counts(case, final)})
        progress[pid] = record
        _write_progress(ctx, progress)
        ctx.set_status(pid, "Completed", reviewer)
        _refresh_csvs(ctx, reviewer)
        st.rerun()


def patient_page(ctx, reviewer: str, pid: str) -> None:
    case = ctx.cases[pid]
    index = ctx.patients.index(pid)
    progress = _progress(ctx)
    if pid not in progress:
        progress[pid] = {"case_order": index + 1, "review_start": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": 0.0}
        _write_progress(ctx, progress)
    st.session_state.setdefault(f"t_last_{pid}", time.time())
    decided = op.decisions(ctx.events.for_patient(pid))
    r = op.resolve(case, decided)
    done = sum(ctx.status(p) == "Completed" for p in ctx.patients)
    head = st.columns([3, 2])
    head[0].markdown(f"## Case {index + 1:02d} of {len(ctx.patients)}")
    head[0].markdown(f"<span class='muted'>Patient study ID: <b>{pid}</b></span>", unsafe_allow_html=True)
    head[1].progress(done / len(ctx.patients), text=f"Progress: {done} / {len(ctx.patients)} patients")
    st.markdown(f"<div class='banner'>{BANNER}</div>", unsafe_allow_html=True)
    with st.expander("Instructions", expanded=index == 0 and not decided):
        st.markdown(TASK)
    bl_vol, fu_vol = volume(ctx, case["baseline_exam"]), volume(ctx, case["followup_exam"])
    all_rows = [f"bl:{b}" for b in r["bl_ids"]] + [f"fu:{f}" for f in r["unclaimed"]]
    focus = _focus_defaults(pid, all_rows)
    if focus and f"z_bl_{pid}" not in st.session_state:
        kind, lesion = focus.split(":", 1)
        if kind == "bl":
            _jump(pid, "bl", bl_vol, lesion)
            _jump(pid, "fu", fu_vol, r["target"].get(lesion) or r["v0"].get(lesion))
        else:
            _jump(pid, "fu", fu_vol, lesion)
    highlight = set()
    if focus:
        kind, lesion = focus.split(":", 1)
        highlight = {lesion, r["target"].get(lesion) if kind == "bl" else r["fu_link"].get(lesion), r["v0"].get(lesion) if kind == "bl" else None}
        pick = st.session_state.get(f"pick_{pid}_{focus}")
        highlight |= {pick} if pick else set()
        highlight.discard(None)
    viewer(pid, bl_vol, fu_vol, highlight)
    tables(ctx, pid, case, r, decided, bl_vol, fu_vol)
    editor(ctx, reviewer, pid, case, r, decided, focus, fu_vol, bl_vol)
    save_block(ctx, reviewer, pid, case, r, decided, index)


def main() -> None:
    st.set_page_config(page_title="OncoState lesion review", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("session_id", "S" + secrets.token_hex(6))
    reviewer = st.session_state.get("reviewer")
    if not reviewer:
        start_screen()
        return
    ctx = _ctx(reviewer)
    _record_order(ctx, reviewer)
    ctx.touch_session(st.session_state["session_id"], reviewer, _current_case(ctx) or "", "review")
    sidebar(ctx, reviewer)
    pid = _current_case(ctx)
    if pid is None:
        _refresh_csvs(ctx, reviewer)
        st.title("Review complete.")
        st.markdown("Thank you. All patients have been reviewed and saved.")
        st.markdown("**Please zip the folder `exports` inside the study folder and send it to the study team.** "
                    f"All your results are also in one spreadsheet: `{_csv_path(ctx, reviewer).relative_to(ROOT).as_posix()}`. "
                    "You may now close this window.")
        st.markdown(f"<div class='banner'>{BANNER}</div>", unsafe_allow_html=True)
        return
    patient_page(ctx, reviewer, pid)


main()
