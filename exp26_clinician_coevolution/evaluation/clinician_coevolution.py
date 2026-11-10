"""OFFLINE co-evolution analysis driven by REAL clinician corrections from the 50-case study.

    python -m evaluation.clinician_coevolution [--exports exports/clinician50] [--out exports/clinician50/results]

verified clinician corrections -> existing candidate repair -> validation gate (non-regression) -> deployed version
-> FUTURE UNSEEN patients. The 50 review patients are the correction source only; validation-gate and future patients
are drawn from the remaining locked Experiment 2.2 patients (development patients excluded) by a fixed seed, written to
configs/batches_clinician50.yaml and checked for overlap before anything is evaluated.

Conditions:  frozen (V0) | memory_only (corrected states kept for the 50 review patients; future model stays V0) |
             naive_update (candidate deployed without validation) | gated_coevolution (deployed only if the gate accepts)
Outputs: clinician_coevolution_results.csv, future_unseen_results.csv, repair_gate_results.csv, correction_efficiency.csv.
Every interval resamples patients. Outputs are labelled SIMULATED when any event was not made by a clinician.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from adapters.schema_mapping import load_bundle
from configs.loader import check_batches, load_config
from corrections.correction_event import is_correction
from corrections.event_store import EventStore
from evaluation.cohorts import ALL_REVIEW_PATIENTS, cohort
from evaluation.gt_access import load_gt
from evaluation.metrics import counts_frame, pair_decisions, summarize
from evaluation.paired_statistics import bootstrap_f1, paired_bootstrap
from repair.build_candidate import InsufficientCorrections, fit_candidate
from repair.matching_repair import CentroidHungarianMatcher
from repair.validation_gate import run_gate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/real/exp22_longitudinal_ct"
BATCHES = ROOT / "configs/batches_clinician50.yaml"
SPLIT_SEED, N_GATE = 20261002, 80
CONDITIONS = ["frozen", "memory_only", "naive_update", "gated_coevolution"]


def make_batches(manifest: dict, review: list[str]) -> dict:
    dev = sorted(p["patient_id"] for p in manifest["patients"] if p["cohort"] != "locked")
    pool = sorted(p["patient_id"] for p in manifest["patients"] if p["cohort"] == "locked" and p["patient_id"] not in review)
    rng = np.random.default_rng(SPLIT_SEED)
    pool = [str(x) for x in rng.permutation(pool)]
    batches = {"development": dev, "correction_batch_1": sorted(review), "validation_gate_1": sorted(pool[:N_GATE]),
               "future_eval_1": sorted(pool[N_GATE:])}
    check_batches(batches)
    assert not set(batches["future_eval_1"]) & set(review) and not set(batches["validation_gate_1"]) & set(review)
    assert not set(dev) & (set(review) | set(pool))
    return batches


def reviewer_events(exports: Path) -> dict[str, list[dict]]:
    """Events of completed patients per reviewer, in the reviewer's review order."""
    out = {}
    for d in sorted(p for p in (exports / "events").iterdir() if p.is_dir()):
        store = EventStore(d, sqlite_mirror=False)
        store.verify()
        status = json.loads((d / "case_status.json").read_text(encoding="utf-8")) if (d / "case_status.json").exists() else {}
        done = {p for p, s in status.items() if s.get("status") == "Completed"}
        out[d.name] = [e for e in store.all() if e["patient_id"] in done]
    return out


def _future(matcher, cases, gt, pids):
    pred = {p: matcher.predict(cases[p])["pairs"] for p in pids}
    return pred, counts_frame(cases, pred, gt, pids)


def _row(condition, events, df, df0, pred, pred0, cases, gt, pids, n_boot, seed, extra=None) -> dict:
    s = summarize(df)
    f, lo, hi = bootstrap_f1(df, n_boot, seed)
    dlt = {m: paired_bootstrap(df, df0, m, n_boot, seed) for m in ("matching_f1", "change_accuracy", "burden_accuracy")}
    a = pair_decisions(cases, pred0, gt, pids)
    b = pair_decisions(cases, pred, gt, pids)
    ga = a[~a.false_pair].set_index(["patient_id", "baseline_lesion_id", "followup_lesion_id"]).correct
    gb = b[~b.false_pair].set_index(["patient_id", "baseline_lesion_id", "followup_lesion_id"]).correct.reindex(ga.index)
    return {"condition": condition, "n_corrections_used": sum(map(is_correction, events)), "n_events_used": len(events),
            "future_n_patients": len(pids), "future_matching_f1": s["matching_f1"], "future_f1_ci_lo": lo, "future_f1_ci_hi": hi,
            "future_matching_precision": s["matching_precision"], "future_matching_recall": s["matching_recall"],
            "future_change_accuracy": s["change_accuracy"], "future_burden_direction_accuracy": s["burden_accuracy"],
            "delta_f1_vs_v0": dlt["matching_f1"]["delta"], "delta_f1_ci_lo": dlt["matching_f1"]["ci_lo"],
            "delta_f1_ci_hi": dlt["matching_f1"]["ci_hi"], "delta_f1_p_boot": dlt["matching_f1"]["p_boot"],
            "delta_change_accuracy_vs_v0": dlt["change_accuracy"]["delta"], "delta_change_ci_lo": dlt["change_accuracy"]["ci_lo"],
            "delta_change_ci_hi": dlt["change_accuracy"]["ci_hi"], "delta_burden_accuracy_vs_v0": dlt["burden_accuracy"]["delta"],
            "delta_burden_ci_lo": dlt["burden_accuracy"]["ci_lo"], "delta_burden_ci_hi": dlt["burden_accuracy"]["ci_hi"],
            "future_errors_fixed": int((~ga & gb).sum()), "future_errors_introduced": int((ga & ~gb).sum()),
            "future_false_pairs": int(b.false_pair.sum()), **(extra or {})}


def run_reviewer(name, events, cases, gt, batches, cfg, n_boot, seed) -> tuple[list[dict], list[dict], list[dict]]:
    v0 = CentroidHungarianMatcher(cfg["v0"]["gate_mm"])
    fut, gate_pids = batches["future_eval_1"], batches["validation_gate_1"]
    pred0, df0 = _future(v0, cases, gt, fut)
    results, gates, eff = [], [], []
    base = {"reviewer_id": name}
    results.append(_row("frozen", [], df0, df0, pred0, pred0, cases, gt, fut, n_boot, seed, {**base, "deployed_version": "V0"}))
    results.append(_row("memory_only", events, df0, df0, pred0, pred0, cases, gt, fut, n_boot, seed,
                        {**base, "deployed_version": "V0", "note": "corrections kept in the 50 review patients' states only"}))
    try:
        cand = fit_candidate(events, cases, cfg, seed)
    except InsufficientCorrections as error:
        cand = None
        gates.append({**base, "candidate": "V1", "accepted": False, "reason": f"no candidate: {error}"})
    if cand is not None:
        predc, dfc = _future(cand, cases, gt, fut)
        results.append(_row("naive_update", events, dfc, df0, predc, pred0, cases, gt, fut, n_boot, seed, {**base, "deployed_version": "V1"}))
        g = run_gate(cases, gt, v0, cand, gate_pids, cfg["non_regression_thresholds"], n_boot, seed,
                     meta={"candidate": "V1", "parent": "V0", "n_training_corrections": sum(map(is_correction, events))})
        gates.append({**base, **{k: v for k, v in g.items() if k not in ("validation_patients", "thresholds")}})
        deployed = cand if g["accepted"] else v0
        predg, dfg = (predc, dfc) if g["accepted"] else (pred0, df0)
        results.append(_row("gated_coevolution", events, dfg, df0, predg, pred0, cases, gt, fut, n_boot, seed,
                            {**base, "deployed_version": "V1" if g["accepted"] else "V0", "gate_accepted": g["accepted"],
                             "gate_reason": g["reason"]}))
        del deployed
    # correction efficiency: cumulative prefixes of the reviewer's patients in review order
    order = list(dict.fromkeys(e["patient_id"] for e in sorted(events, key=lambda e: e["timestamp"])))
    total = sum(map(is_correction, events))
    grid = sorted({0, *[k for k in (10, 20, 40, 60, 80, 100, 120, 160, 200) if k < total], total})
    for k in grid:
        used, n = [], 0
        for p in order:
            if n >= k:
                break
            used += [e for e in events if e["patient_id"] == p]
            n = sum(map(is_correction, used))
        row = {**base, "target_corrections": k, "cumulative_corrections": n, "n_patients_used": len({e["patient_id"] for e in used})}
        if k == 0:
            eff.append({**row, "naive_delta_f1": 0.0, "gated_delta_f1": 0.0, "future_errors_fixed": 0, "future_errors_introduced": 0,
                        "gate_accepted": None})
            continue
        try:
            c = fit_candidate(used, cases, cfg, seed)
        except InsufficientCorrections as error:
            eff.append({**row, "note": str(error)})
            continue
        pc, dc = _future(c, cases, gt, fut)
        r = _row("naive_update", used, dc, df0, pc, pred0, cases, gt, fut, n_boot // 4, seed)
        g = run_gate(cases, gt, v0, c, gate_pids, cfg["non_regression_thresholds"], n_boot // 4, seed)
        eff.append({**row, "naive_delta_f1": r["delta_f1_vs_v0"], "naive_ci_lo": r["delta_f1_ci_lo"], "naive_ci_hi": r["delta_f1_ci_hi"],
                    "gated_delta_f1": r["delta_f1_vs_v0"] if g["accepted"] else 0.0, "gate_accepted": g["accepted"],
                    "future_errors_fixed": r["future_errors_fixed"], "future_errors_introduced": r["future_errors_introduced"],
                    "corrections_per_pp_f1": n / (100 * r["delta_f1_vs_v0"]) if r["delta_f1_vs_v0"] > 0 else None})
    return results, gates, eff


def consensus_events(per_reviewer: dict[str, list[dict]]) -> list[dict]:
    """Majority decision per page row over reviewers (ties -> dropped); the latest event carrying that decision is kept."""
    from app import one_page as op
    votes: dict[tuple, list] = {}
    for evs in per_reviewer.values():
        last = {}
        for e in sorted(evs, key=lambda x: x["timestamp"]):
            last[(e["patient_id"], op.row_id(e))] = e
        for key, e in last.items():
            d = op.decision_of(e)
            votes.setdefault(key, []).append(((d["decision"], d.get("target")), e))
    out = []
    for key, vs in votes.items():
        tally = pd.Series([v for v, _ in vs]).value_counts()
        if len(tally) > 1 and tally.iloc[0] == tally.iloc[1]:
            continue
        winner = tally.index[0]
        out.append(next(e for v, e in reversed(vs) if v == winner))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["50", "30", "10A", "10B", "10C", "10D", "10E"], default="50")
    ap.add_argument("--exports", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", default="configs/experiment_real.yaml")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    c = cohort(a.cohort)
    a.exports = a.exports or c["exports"]
    out = a.out or a.exports / "results"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    bundle = load_bundle(SOURCE)
    cases, gt = bundle["cases"], load_gt(SOURCE)
    # gate / future patients exclude ALL 50 study patients for either cohort, so 30- and 50-case results are comparable
    batches = make_batches(bundle["manifest"], json.loads(ALL_REVIEW_PATIENTS.read_text(encoding="utf-8")))
    review = json.loads((c["secure"] / "selected_ids.json").read_text(encoding="utf-8"))
    BATCHES.write_text(yaml.safe_dump(batches, sort_keys=False), encoding="utf-8")
    per_reviewer = reviewer_events(a.exports)
    if not any(per_reviewer.values()):
        raise SystemExit("no completed clinician reviews yet")
    for evs in per_reviewer.values():
        assert {e["patient_id"] for e in evs} <= set(review), f"events from patients outside the {a.cohort}-case review cohort"
    sets = dict(per_reviewer)
    if len(per_reviewer) > 1:
        sets["CONSENSUS"] = consensus_events(per_reviewer)
    simulated = any(e["correction_source"] != "clinician" for evs in per_reviewer.values() for e in evs)
    results, gates, eff = [], [], []
    for name, evs in sets.items():
        r, g, e = run_reviewer(name, evs, cases, gt, batches, cfg, a.n_boot, cfg["random_seed"])
        results += r
        gates += g
        eff += e
    tag = {"data_source": "SIMULATED" if simulated else "clinician"}
    res = pd.DataFrame(results).assign(**tag)
    keep = ["reviewer_id", "condition", "n_corrections_used", "future_n_patients", "future_matching_f1", "future_matching_precision",
            "future_matching_recall", "future_change_accuracy", "future_burden_direction_accuracy", "delta_f1_vs_v0",
            "delta_f1_ci_lo", "delta_f1_ci_hi", "data_source"]
    res[keep].to_csv(out / "clinician_coevolution_results.csv", index=False)
    res.to_csv(out / "future_unseen_results.csv", index=False)
    pd.DataFrame(gates).assign(**tag).to_csv(out / "repair_gate_results.csv", index=False)
    pd.DataFrame(eff).assign(**tag).to_csv(out / "correction_efficiency.csv", index=False)
    print(f"{tag['data_source']} co-evolution: {len(sets)} correction set(s), future={len(batches['future_eval_1'])} patients, "
          f"gate={len(batches['validation_gate_1'])} patients -> {out}")


if __name__ == "__main__":
    main()
