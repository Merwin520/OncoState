"""Page A — case queue (no reference data shown)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from state.schemas import REVIEW_STATUSES


def queue_table(ctx) -> pd.DataFrame:
    thr = ctx.cfg.get("low_confidence_threshold", 0.4)
    rows = []
    for pid in ctx.patients:
        s = ctx.current_state(pid)
        evs = ctx.events.for_patient(pid)
        conf = [c["confidence"] for c in s["correspondences"] if c["status"] == "persistent" and c.get("confidence") is not None]
        dur = sum(e["review_time_seconds"] or 0 for e in evs)
        rows.append({"patient ID": pid, "batch ID": ctx.batch_of(pid), "model version": s["source_model_version"],
                     "review status": ctx.status(pid), "lesions (BL/FU)": f"{len(s['baseline_lesions'])}/{len(s['followup_lesions'])}",
                     "low-confidence matches": sum(c < thr for c in conf), "review duration (s)": round(dur, 1),
                     "flagged": ctx.status(pid) == "Flagged", "decisions": len(evs)})
    return pd.DataFrame(rows)


def render(ctx) -> None:
    st.subheader("A · Case queue")
    df = queue_table(ctx)
    c1, c2, c3, c4 = st.columns(4)
    batch = c1.selectbox("Batch", ["all"] + sorted(df["batch ID"].unique()), key="q_batch")
    stat = c2.selectbox("Review status", ["all"] + list(REVIEW_STATUSES), key="q_status")
    mv = c3.selectbox("Model version", ["all"] + sorted(df["model version"].unique()), key="q_model")
    low = c4.checkbox("Low-confidence only", key="q_low")
    f = df
    if batch != "all":
        f = f[f["batch ID"] == batch]
    if stat != "all":
        f = f[f["review status"] == stat]
    if mv != "all":
        f = f[f["model version"] == mv]
    if low:
        f = f[f["low-confidence matches"] > 0]
    st.dataframe(f, hide_index=True, width='stretch')
    st.caption("Select a patient in the sidebar to review. Reference annotations are never shown in this interface.")
