"""Page B — real full-volume baseline/follow-up CT review."""
from __future__ import annotations

import streamlit as st

from .ui_utils import load_crop, render_slice, ui_payload, volume

WINDOWS = {
    "Soft tissue": (40.0, 400.0),
    "Lung": (-600.0, 1500.0),
    "Liver": (60.0, 160.0),
    "Bone": (400.0, 1800.0),
    "Custom": (40.0, 400.0),
}


def _initial_slice(v) -> int:
    centroids = v[3]
    return int(round(float(centroids[0, 0]))) if len(centroids) else v[0].shape[0] // 2


def render(ctx, pid: str) -> None:
    st.subheader(f"B · Full-volume longitudinal CT review — {pid}")
    state = ui_payload(ctx.current_state(pid))
    baseline = volume(ctx, state["baseline_exam"])
    followup = volume(ctx, state["followup_exam"])

    controls = st.columns([1, 1, 1, 1, 1])
    show_mask = controls[0].toggle("Predicted mask", value=True, key=f"b_mask_{pid}")
    show_centroid = controls[1].toggle("Predicted centroid", value=True, key=f"b_centroid_{pid}")
    show_ids = controls[2].toggle("Lesion ID", value=True, key=f"b_ids_{pid}")
    preset = controls[3].selectbox("CT window", list(WINDOWS), key=f"b_window_{pid}")
    sync_valid = bool(ctx.cases[pid].get("navigation", {}).get("synchronized_scrolling_valid", False))
    sync = controls[4].toggle("Synchronized", value=False, disabled=not sync_valid, key=f"b_sync_{pid}",
                              help="Enabled only when array geometry is compatible.")
    center_default, width_default = WINDOWS[preset]
    if preset == "Custom":
        wc1, wc2 = st.columns(2)
        center = wc1.number_input("Window center (HU)", -1200.0, 2000.0, center_default, 10.0, key=f"b_wc_{pid}")
        width = wc2.number_input("Window width (HU)", 1.0, 4000.0, width_default, 10.0, key=f"b_ww_{pid}")
    else:
        center, width = center_default, width_default

    left, right = st.columns(2)
    if baseline is not None and followup is not None:
        nb, nf = baseline[0].shape[0], followup[0].shape[0]
        if sync:
            index = st.slider("Axial slice (both exams)", 0, nb - 1, min(_initial_slice(baseline), nb - 1), key=f"b_k_{pid}")
            kb = kf = index
        else:
            kb = left.slider("Baseline axial slice", 0, nb - 1, min(_initial_slice(baseline), nb - 1), key=f"b_kb_{pid}")
            kf = right.slider("Follow-up axial slice", 0, nf - 1, min(_initial_slice(followup), nf - 1), key=f"b_kf_{pid}")
        left.image(render_slice(*baseline, kb, show_mask, show_centroid, show_ids, center, width),
                   caption=f"Baseline · axial {kb + 1}/{nb} · C {center:.0f} / W {width:.0f} HU", width="stretch")
        right.image(render_slice(*followup, kf, show_mask, show_centroid, show_ids, center, width),
                    caption=f"Follow-up · axial {kf + 1}/{nf} · C {center:.0f} / W {width:.0f} HU", width="stretch")
        if not sync_valid:
            st.caption("Baseline and follow-up geometry differs; navigation is independent to avoid misleading slice alignment.")
    else:
        st.error("A required full CT volume is unavailable. This case cannot be reviewed until the pilot bundle is repaired.")

    with st.expander("Optional lesion-centred convenience crops"):
        crop_left, crop_right = st.columns(2)
        for column, key, title in ((crop_left, "baseline_lesions", "Baseline"), (crop_right, "followup_lesions", "Follow-up")):
            column.markdown(f"**{title}**")
            for lesion in state[key]:
                image = load_crop(ctx, lesion.get("crop_ref"))
                if image is not None:
                    column.image(image, caption=f"{lesion['lesion_id']} · {lesion['diameter_mm']:.1f} mm", width=180)

    with st.expander("Predicted lesion table"):
        st.dataframe(
            [{"lesion": lesion["lesion_id"], "side": "baseline", "diameter mm": round(lesion["diameter_mm"], 1),
              "volume mL": round(lesion["volume_mm3"] / 1000, 2)} for lesion in state["baseline_lesions"]] +
            [{"lesion": lesion["lesion_id"], "side": "follow-up", "diameter mm": round(lesion["diameter_mm"], 1),
              "volume mL": round(lesion["volume_mm3"] / 1000, 2)} for lesion in state["followup_lesions"]],
            hide_index=True,
        )
