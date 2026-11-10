"""Regression accounting between a parent version and a candidate on the same patients."""
from __future__ import annotations

import pandas as pd

from evaluation.metrics import pair_decisions


def regressions(cases, pred_parent, pred_cand, gt, pids) -> dict:
    a = pair_decisions(cases, pred_parent, gt, pids)
    b = pair_decisions(cases, pred_cand, gt, pids)
    ga, gb = a[~a.false_pair].set_index(["patient_id", "baseline_lesion_id", "followup_lesion_id"]).correct, \
        b[~b.false_pair].set_index(["patient_id", "baseline_lesion_id", "followup_lesion_id"]).correct
    gb = gb.reindex(ga.index)
    created = int((ga & ~gb).sum())   # previously-correct matches broken by the candidate
    fixed = int((~ga & gb).sum())
    fp_a, fp_b = int(a.false_pair.sum()), int(b.false_pair.sum())
    return {"regressions_created": created, "regressions_fixed": fixed, "false_pairs_parent": fp_a, "false_pairs_candidate": fp_b,
            "broken_patients": sorted(set(ga[ga & ~gb].index.get_level_values(0)))}
