"""Cohort-bundle format used by the whole package, and the mapping from Experiment 2.2 fields.

bundle/
  manifest.json            {"cohort": ..., "data_mode": "real"|"synthetic", "patients": [{"patient_id", "cohort"}], "sha256": ...}
  cases/<pid>.json         clinician-visible case record (lesions, registration ref, crop refs) — NO reference data
  gt_hidden/<pid>.json     hidden reference record — read only via evaluation/gt_access.py
  crops/, volumes/         optional lesion-centred crops (PNG) / downsampled volumes (NPZ) for the UI

Experiment 2.2 -> bundle mapping (adapters/exp22_adapter.py):
  pairs.json[patient_id, baseline_ct, followup_ct, cohort]         -> case.patient_id, *_exam.image_ref, manifest cohort (dev / locked)
  state_A/<pid>.json baseline_lesions{id: volume_mm3, diameter_mm, centroid_mm}  -> case.baseline_lesions[]
  state_A/<pid>.json followup_lesions{...}                          -> case.followup_lesions[]
  reg/<pid>.tfm (SimpleITK, BL->FU)                                 -> case.registration.transform_ref; baseline centroid_registered_mm
  cut_A/<pid>_<tp>.nii.gz (predicted lesion label maps)             -> lesion mask_ref
  exp22b_common.reference(pid, 'A') pred->reference lesion maps     -> gt_hidden pairs / classes / change / burden
"""
from __future__ import annotations

import json
from pathlib import Path

from state.provenance import sha256_json


def load_bundle(bundle_dir: str | Path) -> dict:
    d = Path(bundle_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cases = {}
    for p in sorted((d / "cases").glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        cases[c["patient_id"]] = c
    return {"manifest": manifest, "cases": cases, "root": d}


def write_bundle(bundle_dir: str | Path, cohort: str, data_mode: str, cases: list[dict], gts: list[dict], extra: dict | None = None) -> Path:
    d = Path(bundle_dir)
    (d / "cases").mkdir(parents=True, exist_ok=True)
    (d / "gt_hidden").mkdir(parents=True, exist_ok=True)
    for c in cases:
        (d / "cases" / f"{c['patient_id']}.json").write_text(json.dumps(c, indent=1, sort_keys=True), encoding="utf-8")
    for g in gts:
        (d / "gt_hidden" / f"{g['patient_id']}.json").write_text(json.dumps(g, indent=1, sort_keys=True), encoding="utf-8")
    manifest = {"cohort": cohort, "data_mode": data_mode, "n_patients": len(cases),
                "patients": [{"patient_id": c["patient_id"], "cohort": c.get("cohort", "unknown")} for c in cases],
                "cases_sha256": sha256_json(cases), "gt_sha256": sha256_json(gts), **(extra or {})}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return d
