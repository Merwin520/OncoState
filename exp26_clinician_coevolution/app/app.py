"""OncoState Experiment 2.6 clinician review interface."""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != APP_DIR]
sys.path.insert(0, str(ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st  # noqa: E402

from app import case_queue, matching_review, patient_review, review_summary, secondary_review  # noqa: E402
from app.ui_utils import BANNER, CSS, ReviewContext  # noqa: E402

PAGES = ["A · Case queue", "B · Longitudinal review", "C · Matching review", "D · Secondary review", "E · Review summary"]


def main() -> None:
    st.set_page_config(page_title="OncoState 2.6 review", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='research-banner'>{BANNER}</div>", unsafe_allow_html=True)
    context = ReviewContext.from_config(os.environ.get("EXP26_UI_CONFIG", "configs/ui.yaml"))
    st.session_state.setdefault("session_id", "S" + secrets.token_hex(6))
    with st.sidebar:
        st.markdown("### OncoState · Exp 2.6")
        clinician_id = st.text_input("Study clinician ID", key="clinician_id", help="Study pseudonym only — no names.").strip()
        page = st.radio("Page", PAGES, key="page")
        patient_id = st.selectbox("Patient", context.patients, key="patient")
        st.caption(f"Data: {context.data_mode.upper()} · model {context.model_version} · session {st.session_state['session_id']}")
        if context.data_mode == "synthetic":
            st.caption("SYNTHETIC demonstration data — not patients.")
        else:
            st.caption("10-case de-identified REAL pilot · reference annotations unavailable to this application.")
    context.touch_session(st.session_state["session_id"], clinician_id, patient_id, page)
    if page == PAGES[0]:
        case_queue.render(context)
    elif page == PAGES[1]:
        patient_review.render(context, patient_id)
    elif page == PAGES[2]:
        matching_review.render(context, patient_id, clinician_id)
    elif page == PAGES[3]:
        secondary_review.render(context, patient_id, clinician_id)
    else:
        review_summary.render(context, patient_id, clinician_id)
    st.markdown(f"<div class='small-muted' style='margin-top:24px'>{BANNER}</div>", unsafe_allow_html=True)


main()
