"""Build a candidate repair from verified correction events.

Correction events -> pairwise supervision -> PairwiseLogisticMatcher (candidate, NOT deployed).

Supervision (only from patients whose events are passed in, i.e. correction-source patients):
  accept (a, b)          (a, b) positive; every other candidate of a negative
  correct (a, b0 -> b1)  (a, b1) positive; (a, b0) and every other candidate of a negative
  disappeared a          every candidate of a negative
  new_lesion b           every candidate of b negative
  uncertain              ignored
Clinician identity is never a feature.

CLI:
  python -m repair.build_candidate --config configs/experiment.yaml --events <dir with correction_events.jsonl> \
         --parent V0 --version V1 --batch correction_batch_1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .matching_features import candidate_pairs
from .matching_repair import CentroidHungarianMatcher, PairwiseLogisticMatcher


class InsufficientCorrections(RuntimeError):
    pass


def supervision(events: list[dict], cases: dict[str, dict], v0_gate_mm: float, max_distance_mm: float) -> pd.DataFrame:
    by_pid: dict[str, list[dict]] = {}
    for e in events:
        if e["event_type"] in ("lesion_match", "disappeared_lesion", "new_lesion"):
            by_pid.setdefault(e["patient_id"], []).append(e)
    frames = []
    for pid, evs in by_pid.items():
        case = cases[pid]
        v0 = set(CentroidHungarianMatcher(v0_gate_mm).predict(case)["pairs"])
        cand = candidate_pairs(case, v0, max_distance_mm)
        if cand.empty:
            continue
        label = pd.Series(np.nan, index=cand.index)
        key_a, key_b = cand.baseline_lesion_id, cand.followup_lesion_id
        for e in sorted(evs, key=lambda x: x["timestamp"]):  # later decisions on the same item win
            a, b0, b1 = e["baseline_lesion_id"], e["original_followup_lesion_id"], e["corrected_followup_lesion_id"]
            if e["action"] == "accept" and a is not None and b0 is not None:
                label[(key_a == a)] = 0.0
                label[(key_a == a) & (key_b == b0)] = 1.0
            elif e["action"] == "correct" and a is not None and b1 is not None:
                label[(key_a == a)] = 0.0
                label[key_b == b1] = 0.0
                label[(key_a == a) & (key_b == b1)] = 1.0
            elif e["action"] == "disappeared" and a is not None:
                label[key_a == a] = 0.0
            elif e["action"] == "new_lesion":
                b = b0 if b0 is not None else b1
                if b is not None:
                    label[key_b == b] = 0.0
        cand = cand.assign(label=label, source_event_patient=pid).dropna(subset=["label"])
        frames.append(cand)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_candidate(events: list[dict], cases: dict[str, dict], cfg: dict, seed: int) -> PairwiseLogisticMatcher:
    rc = cfg["repair_model"]
    sup = supervision(events, cases, cfg["v0"]["gate_mm"], rc["max_distance_mm"])
    if sup.empty or sup.label.nunique() < 2 or min((sup.label == 1).sum(), (sup.label == 0).sum()) < rc["min_examples_per_class"]:
        raise InsufficientCorrections(f"need >= {rc['min_examples_per_class']} positive and negative verified pairs "
                                      f"(have {int((sup.label == 1).sum()) if not sup.empty else 0} / {int((sup.label == 0).sum()) if not sup.empty else 0})")
    feats = cfg["matching_features"]
    X = sup[feats].to_numpy(float)
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    lr = LogisticRegression(C=rc["C"], class_weight=rc.get("class_weight"), max_iter=5000, random_state=seed)
    lr.fit((X - mean) / scale, sup.label.astype(int))
    return PairwiseLogisticMatcher(feats, mean, scale, lr.coef_[0], lr.intercept_[0], rc["threshold"], rc["max_distance_mm"],
                                   cfg["v0"]["gate_mm"], training={"n_pairs": int(len(sup)), "n_pos": int((sup.label == 1).sum()),
                                                                   "n_neg": int((sup.label == 0).sum()), "n_patients": int(sup.source_event_patient.nunique()),
                                                                   "n_events": len(events), "seed": seed})


def main():
    from configs.loader import load_config
    from corrections.event_store import EventStore
    from adapters.schema_mapping import load_bundle
    from .version_manager import VersionManager

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--events", required=True, help="directory containing correction_events.jsonl")
    ap.add_argument("--parent", default="V0")
    ap.add_argument("--version", required=True)
    ap.add_argument("--batch", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    cases = load_bundle(cfg["paths"]["bundle"])["cases"]
    events = EventStore(a.events, sqlite_mirror=False).all()
    m = fit_candidate(events, cases, cfg, cfg["random_seed"])
    vm = VersionManager(cfg["paths"]["versions"])
    path = vm.register_candidate(a.version, a.parent, m.to_dict(), correction_batch=a.batch, events=events, config=cfg)
    print(f"candidate {a.version} (parent {a.parent}) written to {path}; NOT deployed until the validation gate accepts it")


if __name__ == "__main__":
    main()
