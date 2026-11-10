"""Deterministic offline metrics against hidden reference data.

Matching is evaluated at the level of PREDICTED lesions (the lesions the segmentation produced):
  GT pair       (a, b) of predicted baseline/follow-up lesions that map to the same reference lesion
  precision     correct predicted pairs / predicted pairs
  recall        correct predicted pairs / GT pairs
  F1            2 TP / (predicted pairs + GT pairs)   (pooled over patients = ratio of sums)
Lesion change: among correctly matched pairs, predicted change class (+-20 % volume) == reference change class.
Burden direction: patient-level direction of total predicted volume == reference direction.
New / disappeared: per reference-new follow-up lesion (resp. reference-disappeared baseline lesion), whether it was
left unmatched by the prediction; false-new / false-disappeared = lesions of a GT pair that were left unmatched.
All quantities are per-patient counts so they pool and bootstrap exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from state.patient_state import burden_direction, change_class

COUNT_COLS = ["tp", "pred_pairs", "gt_pairs", "chg_ok", "chg_n", "burden_ok", "new_ok", "new_n", "dis_ok", "dis_n",
              "false_new", "false_dis", "n_fu", "n_bl"]


def patient_counts(case: dict, pairs: list[tuple[str, str]], gt: dict) -> dict:
    pairs = set(map(tuple, pairs))
    bl = {l["lesion_id"]: l for l in case["baseline_lesions"]}
    fu = {l["lesion_id"]: l for l in case["followup_lesions"]}
    tp_pairs = pairs & gt["pairs"]
    mb, mf = {a for a, _ in pairs}, {b for _, b in pairs}
    chg_ok = sum(change_class(bl[a]["volume_mm3"], fu[b]["volume_mm3"]) == gt["change"].get((a, b)) for a, b in tp_pairs
                 if gt["change"].get((a, b)) is not None)
    chg_n = sum(gt["change"].get(p) is not None for p in tp_pairs)
    gt_new = [b for b, c in gt["followup_class"].items() if c == "newly_appearing"]
    gt_dis = [a for a, c in gt["baseline_class"].items() if c == "disappearing"]
    gt_pb, gt_pf = {a for a, _ in gt["pairs"]}, {b for _, b in gt["pairs"]}
    tb = sum(l["volume_mm3"] for l in bl.values())
    tf = sum(l["volume_mm3"] for l in fu.values())
    return {
        "patient_id": case["patient_id"], "tp": len(tp_pairs), "pred_pairs": len(pairs), "gt_pairs": len(gt["pairs"]),
        "chg_ok": int(chg_ok), "chg_n": int(chg_n),
        "burden_ok": int(gt.get("burden") is not None and burden_direction(tb, tf) == gt["burden"]),
        "new_ok": sum(b not in mf for b in gt_new), "new_n": len(gt_new),
        "dis_ok": sum(a not in mb for a in gt_dis), "dis_n": len(gt_dis),
        "false_new": sum(b not in mf for b in gt_pf), "false_dis": sum(a not in mb for a in gt_pb),
        "n_fu": len(fu), "n_bl": len(bl),
    }


def counts_frame(cases: dict, predictions: dict, gt: dict, pids) -> pd.DataFrame:
    return pd.DataFrame([patient_counts(cases[p], predictions[p], gt[p]) for p in sorted(pids)])


def _safe(a, b):
    return float(a / b) if b else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    s = df[COUNT_COLS].sum()
    return {
        "n_patients": int(len(df)), "matching_f1": _safe(2 * s.tp, s.pred_pairs + s.gt_pairs),
        "matching_precision": _safe(s.tp, s.pred_pairs), "matching_recall": _safe(s.tp, s.gt_pairs),
        "change_accuracy": _safe(s.chg_ok, s.chg_n), "burden_accuracy": _safe(s.burden_ok, len(df)),
        "new_lesion_accuracy": _safe(s.new_ok, s.new_n), "disappeared_lesion_accuracy": _safe(s.dis_ok, s.dis_n),
        "false_new_rate": _safe(s.false_new, s.n_fu), "false_disappeared_rate": _safe(s.false_dis, s.n_bl),
        "matching_errors": int(s.pred_pairs - s.tp + s.gt_pairs - s.tp),
    }


def pair_decisions(cases: dict, predictions: dict, gt: dict, pids) -> pd.DataFrame:
    """One row per GT pair: was it recovered? (input for regressions created / fixed)."""
    rows = []
    for p in sorted(pids):
        pr = set(map(tuple, predictions[p]))
        for a, b in sorted(gt[p]["pairs"]):
            rows.append({"patient_id": p, "baseline_lesion_id": a, "followup_lesion_id": b, "correct": (a, b) in pr})
        for a, b in sorted(pr - gt[p]["pairs"]):
            rows.append({"patient_id": p, "baseline_lesion_id": a, "followup_lesion_id": b, "correct": False, "false_pair": True})
    df = pd.DataFrame(rows)
    if "false_pair" not in df:
        df["false_pair"] = False
    df["false_pair"] = df["false_pair"].fillna(False).astype(bool)
    return df
