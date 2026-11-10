"""Page E — review summary and case completion."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from corrections.correction_event import is_correction


def summary(events: list[dict]) -> dict:
    cnt = lambda f: sum(1 for e in events if f(e))  # noqa: E731
    return {"accepted matches": cnt(lambda e: e["action"] == "accept"),
            "corrected matches": cnt(lambda e: e["action"] == "correct"),
            "disappeared lesions": cnt(lambda e: e["action"] == "disappeared"),
            "new lesions": cnt(lambda e: e["action"] == "new_lesion"),
            "uncertain decisions": cnt(lambda e: e["action"] == "uncertain"),
            "measurement corrections": cnt(lambda e: e["action"] == "measurement"),
            "segmentation flagged incorrect": cnt(lambda e: e["action"] == "segmentation_incorrect"),
            "verified corrections (changed the model's answer)": cnt(is_correction),
            "review duration (s)": round(sum(e["review_time_seconds"] or 0 for e in events), 1)}


def render(ctx, pid: str, clinician_id: str) -> None:
    from .matching_review import open_items
    st.subheader(f"E · Review summary — {pid}")
    evs = ctx.events.for_patient(pid)
    st.table(pd.DataFrame(summary(evs).items(), columns=["", "count"]))
    s = ctx.current_state(pid)
    st.caption(f"Original prediction: state v0 (unchanged). Current corrected state: v{s['state_version']} "
               f"after {len(s['correction_history'])} recorded decisions. Model parameters are not changed by this review.")
    remaining = len(open_items(ctx, pid))
    if remaining:
        st.warning(f"{remaining} matching item(s) still without a decision.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Mark completed", key="e_done", disabled=bool(remaining) or not clinician_id):
        ctx.set_status(pid, "Completed", clinician_id)
        st.success("Saved: Completed")
    if c2.button("Flag case", key="e_flag", disabled=not clinician_id):
        ctx.set_status(pid, "Flagged", clinician_id)
        st.info("Saved: Flagged")
    if c3.button("Exclude case", key="e_excl", disabled=not clinician_id):
        ctx.set_status(pid, "Excluded", clinician_id)
        st.info("Saved: Excluded")
    st.caption(f"Status: {ctx.status(pid)}")
