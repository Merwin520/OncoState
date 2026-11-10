"""Independent validation gate with non-regression rules.

A candidate is evaluated against the CURRENTLY DEPLOYED parent on validation-gate patients that were never correction
sources and are never future-evaluation patients for this candidate. Thresholds are pre-specified in
configs/experiment.yaml (non_regression_thresholds) and copied into every gate record.

Accept iff ALL hold:
  1. primary   delta matching F1 (candidate - parent) >= min_delta_f1  AND  its patient-bootstrap lower CI >= -noninferiority_margin
  2. safety    precision, recall, change accuracy, burden accuracy drop by no more than max_metric_drop
               false-new and false-disappeared rates rise by no more than max_rate_increase
  3. regressions_created <= regressions_fixed + max_net_regressions
"""
from __future__ import annotations

import argparse

from evaluation.metrics import counts_frame, summarize
from evaluation.paired_statistics import paired_bootstrap
from .non_regression import regressions

SAFETY_HIGHER = ("matching_precision", "matching_recall", "change_accuracy", "burden_accuracy")
SAFETY_LOWER = ("false_new_rate", "false_disappeared_rate")


def run_gate(cases, gt, parent_matcher, cand_matcher, pids, thresholds: dict, n_boot: int, seed: int,
             meta: dict | None = None) -> dict:
    pids = sorted(pids)
    pp = {p: parent_matcher.predict(cases[p])["pairs"] for p in pids}
    pc = {p: cand_matcher.predict(cases[p])["pairs"] for p in pids}
    dp, dc = counts_frame(cases, pp, gt, pids), counts_frame(cases, pc, gt, pids)
    sp, sc = summarize(dp), summarize(dc)
    boot = paired_bootstrap(dc, dp, "matching_f1", n_boot, seed)
    reg = regressions(cases, pp, pc, gt, pids)
    reasons = []
    if not boot["delta"] >= thresholds["min_delta_f1"]:
        reasons.append(f"delta F1 {boot['delta']:+.4f} < {thresholds['min_delta_f1']}")
    if not boot["ci_lo"] >= -thresholds["noninferiority_margin"]:
        reasons.append(f"F1 lower CI {boot['ci_lo']:+.4f} < -{thresholds['noninferiority_margin']}")
    for k in SAFETY_HIGHER:
        if sp[k] == sp[k] and sc[k] == sc[k] and sc[k] < sp[k] - thresholds["max_metric_drop"]:
            reasons.append(f"{k} dropped {sc[k] - sp[k]:+.4f}")
    for k in SAFETY_LOWER:
        if sp[k] == sp[k] and sc[k] == sc[k] and sc[k] > sp[k] + thresholds["max_rate_increase"]:
            reasons.append(f"{k} rose {sc[k] - sp[k]:+.4f}")
    if reg["regressions_created"] > reg["regressions_fixed"] + thresholds["max_net_regressions"]:
        reasons.append(f"regressions created {reg['regressions_created']} > fixed {reg['regressions_fixed']} + {thresholds['max_net_regressions']}")
    accepted = not reasons
    return {**(meta or {}), "validation_patients": pids, "n_validation_patients": len(pids),
            "validation_delta_f1": boot["delta"], "validation_ci_lo": boot["ci_lo"], "validation_ci_hi": boot["ci_hi"],
            "validation_p_boot": boot["p_boot"], "parent_metrics": sp, "candidate_metrics": sc,
            "delta_matching_f1": sc["matching_f1"] - sp["matching_f1"], "delta_change_accuracy": sc["change_accuracy"] - sp["change_accuracy"],
            "delta_burden_accuracy": sc["burden_accuracy"] - sp["burden_accuracy"], **{k: v for k, v in reg.items() if k != "broken_patients"},
            "non_regression_pass": not any(r for r in reasons if not r.startswith("delta F1") and not r.startswith("F1 lower")),
            "accepted": accepted, "reason": "accepted" if accepted else "; ".join(reasons), "thresholds": thresholds}


def main():
    from configs.loader import load_config, load_batches
    from adapters.schema_mapping import load_bundle
    from evaluation.gt_access import load_gt
    from .matching_repair import matcher_from_dict
    from .version_manager import VersionManager

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--gate-batch", required=True, help="batch name in the batch manifest, e.g. validation_gate_1")
    ap.add_argument("--deploy-if-accepted", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    bundle = load_bundle(cfg["paths"]["bundle"])
    gt = load_gt(cfg["paths"]["bundle"])
    batches = load_batches(cfg)
    vm = VersionManager(cfg["paths"]["versions"])
    parent = vm.manifest(a.candidate)["parent_version"]
    res = run_gate(bundle["cases"], gt, matcher_from_dict(vm.model(parent)), matcher_from_dict(vm.model(a.candidate)),
                   batches[a.gate_batch], cfg["non_regression_thresholds"], cfg["bootstrap_iterations"], cfg["random_seed"],
                   meta={"candidate_version": a.candidate, "parent_version": parent, "gate_batch": a.gate_batch,
                         "correction_batch": vm.manifest(a.candidate)["correction_batch"]})
    vm.record_gate(a.candidate, res)
    print(f"{a.candidate}: {'ACCEPTED' if res['accepted'] else 'REJECTED'} — {res['reason']} "
          f"(delta F1 {res['validation_delta_f1']:+.4f} [{res['validation_ci_lo']:+.4f}, {res['validation_ci_hi']:+.4f}])")
    if res["accepted"] and a.deploy_if_accepted:
        vm.deploy(a.candidate)
        print(f"deployed {a.candidate}")


if __name__ == "__main__":
    main()
