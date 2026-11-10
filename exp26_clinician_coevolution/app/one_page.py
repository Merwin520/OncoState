"""One-patient-one-page review logic (no Streamlit, no reference data).

The page shows every V0 proposal of a patient at once:
  * one row per baseline lesion   ("bl:<B>")  — V0 partner F, or V0 "disappeared"
  * one row per follow-up lesion that is not claimed by any baseline row in the clinician's CURRENT mapping ("fu:<F>")
    — V0 "new", or released by one of the clinician's own changes.
Rows default to the V0 proposal. A clinician edit is one immutable correction event (existing schema); on Save Patient
every row the clinician left untouched gets an explicit acceptance event whose review_item ends in "@save", so
"actively changed" and "left as proposed" stay distinguishable. The latest event per row is the row's decision.
"""
from __future__ import annotations

import re

IMPLICIT = "@save"
DECISIONS = ("keep", "changed_match", "disappeared", "new", "match_baseline", "uncertain")


def natural(lesion_id: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(lesion_id))]


def v0_pairs(case: dict) -> dict[str, str]:
    """Frozen V0 baseline -> follow-up proposals serialised in the clinician bundle."""
    return {a: b for a, b in case["v0_prediction"]["pairs"]}


def row_id(event: dict) -> str:
    return event["review_item"].removesuffix(IMPLICIT)


def decision_of(event: dict) -> dict:
    """Translate a stored event back into a row decision."""
    rid, action = row_id(event), event["action"]
    if rid.startswith("fu:"):
        if action in ("correct", "accept"):
            return {"decision": "match_baseline", "target": event["baseline_lesion_id"]}
        return {"decision": {"new_lesion": "new", "uncertain": "uncertain"}.get(action, "new"), "target": None}
    if action == "correct":
        return {"decision": "changed_match", "target": event["corrected_followup_lesion_id"]}
    if action == "accept":
        return {"decision": "keep", "target": None}
    if action == "disappeared":
        keep = event["original_followup_lesion_id"] is None
        return {"decision": "keep" if keep else "disappeared", "target": None}
    return {"decision": "uncertain", "target": None}


def decisions(events: list[dict]) -> dict[str, dict]:
    out = {}
    for e in sorted(events, key=lambda x: x["timestamp"]):
        if e["review_item"].startswith(("bl:", "fu:")):
            out[row_id(e)] = {**decision_of(e), "implicit": e["review_item"].endswith(IMPLICIT)}
    return out


def resolve(case: dict, decided: dict[str, dict]) -> dict:
    """Clinician's current mapping and the rows to display. Never reads reference data."""
    v0 = v0_pairs(case)
    bl_ids = sorted((l["lesion_id"] for l in case["baseline_lesions"]), key=natural)
    fu_ids = sorted((l["lesion_id"] for l in case["followup_lesions"]), key=natural)
    target = {}
    for b in bl_ids:
        d = decided.get(f"bl:{b}")
        if d is None or d["decision"] == "keep":
            target[b] = v0.get(b)
        elif d["decision"] == "changed_match":
            target[b] = d["target"]
        else:
            target[b] = None
    claims: dict[str, list[str]] = {}
    for b, f in target.items():
        if f is not None:
            claims.setdefault(f, []).append(b)
    unclaimed = [f for f in fu_ids if f not in claims]
    fu_link = {}
    for f in unclaimed:
        d = decided.get(f"fu:{f}")
        if d and d["decision"] == "match_baseline" and d["target"]:
            fu_link[f] = d["target"]
    conflicts = {f: sorted(bs, key=natural) for f, bs in claims.items() if len(bs) > 1}
    used_by_fu: dict[str, list[str]] = {}
    for f, b in fu_link.items():
        used_by_fu.setdefault(b, []).append(f)
    for b, fs in used_by_fu.items():  # a baseline lesion linked from the follow-up side may not already have a partner
        if target.get(b) is not None or len(fs) > 1:
            for f in fs:
                conflicts[f] = sorted({b, *conflicts.get(f, [])}, key=natural)
    pairs = sorted({(b, f) for b, f in target.items() if f} | {(b, f) for f, b in fu_link.items()}, key=lambda p: natural(p[0]))
    return {"v0": v0, "bl_ids": bl_ids, "fu_ids": fu_ids, "target": target, "unclaimed": unclaimed, "fu_link": fu_link,
            "conflicts": conflicts, "pairs": pairs}


def counts(case: dict, decided: dict[str, dict]) -> dict:
    r = resolve(case, decided)
    n = {"unchanged": 0, "changed": 0, "disappeared": 0, "new": 0, "uncertain": 0}
    for b in r["bl_ids"]:
        d = decided.get(f"bl:{b}", {"decision": "keep"})
        if d["decision"] == "keep":
            n["unchanged"] += 1
        elif d["decision"] == "changed_match":
            n["changed"] += 1
        else:
            n[d["decision"]] += 1
    for f in r["unclaimed"]:
        d = decided.get(f"fu:{f}", {"decision": "new"})
        if d["decision"] == "match_baseline":
            n["changed"] += 1
        elif d["decision"] == "uncertain":
            n["uncertain"] += 1
        else:
            n["new"] += 1
    return n


def event_for(case: dict, rid: str, decision: str, target: str | None) -> dict:
    """Arguments for corrections.correction_event.clinician_event (existing schema) for one row decision."""
    v0 = v0_pairs(case)
    kind, lesion = rid.split(":", 1)
    if kind == "bl":
        ai = v0.get(lesion)
        if decision == "changed_match" and target == ai:
            decision = "keep"
        action = {"keep": "accept" if ai else "disappeared", "changed_match": "correct", "disappeared": "disappeared",
                  "uncertain": "uncertain"}[decision]
        return {"action": action, "baseline_lesion_id": lesion, "original_followup_lesion_id": ai,
                "corrected_followup_lesion_id": target if action == "correct" else None,
                "original_prediction": {"kind": "pair" if ai else "disappeared", "baseline_lesion_id": lesion, "followup_lesion_id": ai},
                "corrected_value": {"baseline_lesion_id": lesion, "followup_lesion_id": target} if action == "correct" else None}
    if decision == "match_baseline":
        ai_of_target = v0.get(target)
        return {"action": "correct" if ai_of_target != lesion else "accept", "baseline_lesion_id": target,
                "original_followup_lesion_id": ai_of_target,
                "corrected_followup_lesion_id": lesion if ai_of_target != lesion else None,
                "original_prediction": {"kind": "new", "baseline_lesion_id": None, "followup_lesion_id": lesion},
                "corrected_value": {"baseline_lesion_id": target, "followup_lesion_id": lesion}}
    action = {"new": "new_lesion", "uncertain": "uncertain"}[decision]
    return {"action": action, "baseline_lesion_id": None, "original_followup_lesion_id": lesion, "corrected_followup_lesion_id": None,
            "original_prediction": {"kind": "new", "baseline_lesion_id": None, "followup_lesion_id": lesion}, "corrected_value": None}


def implicit_rows(case: dict, decided: dict[str, dict]) -> list[str]:
    """Rows still on their V0 default at save time (each receives an explicit '@save' acceptance event)."""
    r = resolve(case, decided)
    return [f"bl:{b}" for b in r["bl_ids"] if f"bl:{b}" not in decided] + \
           [f"fu:{f}" for f in r["unclaimed"] if f"fu:{f}" not in decided]
