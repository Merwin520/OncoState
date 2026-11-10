"""Model-version registry: candidates are registered, the gate decides, only accepted versions can be deployed.

versions/
  registry.json                 {"deployed": "V1", "versions": {...}, "history": [...]}
  V0/model.json, V0/manifest.json
  V1/model.json, V1/manifest.json, V1/gate.json (after the gate)

A correction changes a PATIENT STATE (state/state_versioning.py). Only this module changes the MATCHING MODEL, and only
after an explicit gate decision.
"""
from __future__ import annotations

import json
from pathlib import Path

from state.provenance import environment, now_iso, sha256_json


class DeploymentRefused(RuntimeError):
    pass


class VersionManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.reg_path = self.root / "registry.json"
        if not self.reg_path.exists():
            self._save({"deployed": None, "versions": {}, "history": []})

    def _load(self) -> dict:
        return json.loads(self.reg_path.read_text())

    def _save(self, reg: dict) -> None:
        tmp = self.reg_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=1, sort_keys=True))
        tmp.replace(self.reg_path)

    @property
    def deployed(self) -> str | None:
        return self._load()["deployed"]

    def model(self, version: str) -> dict:
        return json.loads((self.root / version / "model.json").read_text())

    def manifest(self, version: str) -> dict:
        return json.loads((self.root / version / "manifest.json").read_text())

    def status(self, version: str) -> str:
        return self._load()["versions"][version]["status"]

    def register_base(self, version: str, model: dict, config: dict, data_manifest_sha256: str | None = None) -> Path:
        """Register the frozen source model (V0) and deploy it."""
        d = self._write(version, None, model, "frozen source model", None, [], config, data_manifest_sha256)
        reg = self._load()
        reg["versions"][version] = {"status": "accepted", "parent": None}
        reg["deployed"] = version
        reg["history"].append({"version": version, "parent_version": None, "event": "deployed", "timestamp": now_iso(),
                               "correction_batch": None, "repair_type": model["type"], "validation_metric": None})
        self._save(reg)
        return d

    def register_candidate(self, version: str, parent: str, model: dict, correction_batch: str, events: list[dict], config: dict,
                           data_manifest_sha256: str | None = None) -> Path:
        reg = self._load()
        if parent not in reg["versions"]:
            raise KeyError(f"unknown parent version {parent}")
        if version in reg["versions"]:
            raise FileExistsError(f"version {version} already registered")
        d = self._write(version, parent, model, "candidate repair", correction_batch, events, config, data_manifest_sha256)
        reg["versions"][version] = {"status": "candidate", "parent": parent}
        reg["history"].append({"version": version, "parent_version": parent, "event": "candidate", "timestamp": now_iso(),
                               "correction_batch": correction_batch, "repair_type": model["type"], "validation_metric": None})
        self._save(reg)
        return d

    def _write(self, version, parent, model, role, batch, events, config, data_sha) -> Path:
        d = self.root / version
        d.mkdir(parents=True, exist_ok=False)
        (d / "model.json").write_text(json.dumps(model, indent=1, sort_keys=True))
        sources = sorted({e["correction_source"] for e in events})
        manifest = {"version": version, "parent_version": parent, "role": role, "created": now_iso(),
                    "model_sha256": sha256_json(model), "correction_batch": batch, "n_events": len(events),
                    "correction_sources": sources, "correction_patients": sorted({e["patient_id"] for e in events}),
                    "correction_events_sha256": sha256_json(events) if events else None, "data_manifest_sha256": data_sha,
                    "random_seed": config.get("random_seed"), "config_sha256": sha256_json(config), "environment": environment()}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        return d

    def record_gate(self, version: str, gate: dict) -> None:
        reg = self._load()
        if reg["versions"][version]["status"] != "candidate":
            raise RuntimeError(f"{version} already decided")
        (self.root / version / "gate.json").write_text(json.dumps(gate, indent=1, sort_keys=True))
        reg["versions"][version]["status"] = "accepted" if gate["accepted"] else "rejected"
        reg["history"].append({"version": version, "parent_version": reg["versions"][version]["parent"],
                               "event": "accepted" if gate["accepted"] else "rejected", "timestamp": now_iso(),
                               "correction_batch": gate.get("correction_batch"), "repair_type": self.model(version)["type"],
                               "validation_metric": gate.get("validation_delta_f1"), "reason": gate.get("reason")})
        self._save(reg)

    def deploy(self, version: str) -> None:
        reg = self._load()
        st = reg["versions"].get(version, {}).get("status")
        if st != "accepted":
            raise DeploymentRefused(f"{version} is '{st}'; only gate-accepted versions can be deployed")
        reg["deployed"] = version
        reg["history"].append({"version": version, "parent_version": reg["versions"][version]["parent"], "event": "deployed",
                               "timestamp": now_iso(), "correction_batch": None, "repair_type": self.model(version)["type"],
                               "validation_metric": None})
        self._save(reg)

    def history(self) -> list[dict]:
        return self._load()["history"]


def main():
    """python -m repair.version_manager init|show --config configs/experiment.yaml"""
    import argparse
    from configs.loader import load_config
    from .matching_repair import CentroidHungarianMatcher
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "show"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    vm = VersionManager(cfg["paths"]["versions"])
    if a.cmd == "init":
        if "V0" in vm._load()["versions"]:
            print("V0 already registered")
        else:
            vm.register_base("V0", CentroidHungarianMatcher(cfg["v0"]["gate_mm"]).to_dict(), cfg)
            print(f"registered and deployed frozen V0 in {vm.root}")
    print(json.dumps({"deployed": vm.deployed, "history": vm.history()}, indent=1))


if __name__ == "__main__":
    main()
