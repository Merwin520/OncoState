"""Per-reviewer CSV tables from the reviewer's own folder. Reads NO reference data; packaged with the review app.

Called after every saved patient, so the reviewer's folder always holds ONE up-to-date, spreadsheet-readable file:
    exports/clinician50/<reviewer>_review_results.csv
one row per lesion decision of every SAVED patient (in review order), with that patient's review time and comment.
The hash-chained event log (events/<reviewer>/correction_events.jsonl) remains the source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app import one_page as op
from corrections.event_store import EventStore


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def reviewer_tables(reviewer_dir: Path, cases: dict) -> tuple[list[dict], list[dict], dict]:
    reviewer = reviewer_dir.name
    store = EventStore(reviewer_dir, sqlite_mirror=False)
    n_events = store.verify()  # raises TamperedLogError if the returned log was edited
    events = store.all()
    progress = _read(reviewer_dir / "patient_progress.json", {})
    status = _read(reviewer_dir / "case_status.json", {})
    order = _read(reviewer_dir / "case_order.json", {}).get("case_order", [])
    rows, patients = [], []
    for pid in order or sorted({e["patient_id"] for e in events}):
        case = cases[pid]
        evs = [e for e in events if e["patient_id"] == pid]
        decided = op.decisions(evs)
        res = op.resolve(case, decided)
        completed = status.get(pid, {}).get("status") == "Completed"
        last = {}
        edits = {}
        for e in sorted(evs, key=lambda x: x["timestamp"]):
            rid = op.row_id(e)
            last[rid] = e
            if not e["review_item"].endswith(op.IMPLICIT):
                edits[rid] = edits.get(rid, 0) + 1
        for rid, e in last.items():
            d = decided[rid]
            kind, lesion = rid.split(":", 1)
            if kind == "bl":
                ai = res["v0"].get(lesion)
                decision = d["decision"]
                rows.append({"reviewer_id": reviewer, "patient_id": pid, "case_order": order.index(pid) + 1 if pid in order else None,
                             "row_type": "prior_lesion", "baseline_lesion_id": lesion, "V0_followup_lesion_id": ai,
                             "clinician_decision": decision,
                             "clinician_followup_lesion_id": res["target"][lesion] if decision in ("keep", "changed_match") else None,
                             "decision_mode": "left_as_proposed" if d["implicit"] else "explicit", "n_edits": edits.get(rid, 0),
                             "review_seconds": sum(x["review_time_seconds"] or 0 for x in evs if op.row_id(x) == rid),
                             "timestamp": e["timestamp"], "patient_completed": completed})
            else:
                if lesion not in res["unclaimed"]:
                    continue  # a decision on a follow-up row that the reviewer later linked from the prior side
                decision = {"match_baseline": "changed_match"}.get(d["decision"], d["decision"])
                if decision == "new" and lesion not in res["v0"].values():
                    decision = "keep"  # V0 said new and the reviewer agreed
                rows.append({"reviewer_id": reviewer, "patient_id": pid, "case_order": order.index(pid) + 1 if pid in order else None,
                             "row_type": "unmatched_followup_lesion", "baseline_lesion_id": d["target"],
                             "V0_followup_lesion_id": None, "clinician_decision": decision, "clinician_followup_lesion_id": lesion,
                             "decision_mode": "left_as_proposed" if d["implicit"] else "explicit", "n_edits": edits.get(rid, 0),
                             "review_seconds": sum(x["review_time_seconds"] or 0 for x in evs if op.row_id(x) == rid),
                             "timestamp": e["timestamp"], "patient_completed": completed})
        n = op.counts(case, decided)
        patients.append({"reviewer_id": reviewer, "patient_id": pid, "case_order": order.index(pid) + 1 if pid in order else None,
                         "completed": completed, "n_baseline_lesions": len(case["baseline_lesions"]),
                         "n_followup_lesions": len(case["followup_lesions"]), "n_V0_pairs": len(res["v0"]),
                         "n_unchanged": n["unchanged"], "n_changed": n["changed"], "n_disappeared": n["disappeared"],
                         "n_new": n["new"], "n_uncertain": n["uncertain"],
                         "patient_review_seconds": progress.get(pid, {}).get("seconds"),
                         "review_start": progress.get(pid, {}).get("review_start"), "review_end": progress.get(pid, {}).get("review_end"),
                         "comment": progress.get(pid, {}).get("comment")})
    done = [p for p in patients if p["completed"]]
    current = next((p for p in order if status.get(p, {}).get("status") != "Completed"), None)
    summary = {"reviewer_id": reviewer, "patients_completed": len(done), "current_patient": current,
               "total_review_seconds": round(sum(p["patient_review_seconds"] or 0 for p in done), 1), "n_events": n_events}
    return rows, patients, summary


RESULT_COLUMNS = ["reviewer_id", "case_order", "patient_id", "row_type", "baseline_lesion_id", "V0_followup_lesion_id",
                  "clinician_decision", "clinician_followup_lesion_id", "decision_mode", "n_edits", "row_review_seconds",
                  "patient_review_seconds", "patient_comment", "saved_at"]


def results_frame(reviewer_dir: Path, cases: dict) -> pd.DataFrame:
    rows, patients, _ = reviewer_tables(reviewer_dir, cases)
    done = {p["patient_id"]: p for p in patients if p["completed"]}
    out = [{**r, "row_review_seconds": r["review_seconds"], "patient_review_seconds": done[r["patient_id"]]["patient_review_seconds"],
            "patient_comment": done[r["patient_id"]]["comment"], "saved_at": done[r["patient_id"]]["review_end"]}
           for r in rows if r["patient_id"] in done]
    frame = pd.DataFrame(out, columns=RESULT_COLUMNS)
    return frame.sort_values(["case_order", "row_type", "baseline_lesion_id"], kind="stable") if len(frame) else frame


def write_reviewer_csv(reviewer_dir: Path, cases: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_frame(reviewer_dir, cases).to_csv(path, index=False, encoding="utf-8-sig")
