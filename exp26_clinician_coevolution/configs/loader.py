"""Config loading. Paths in configs are relative to the package root."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path = "configs/experiment.yaml", overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    for k, v in (overrides or {}).items():
        cfg[k] = v if not isinstance(v, dict) else {**cfg.get(k, {}), **v}
    cfg["paths"] = {k: str(resolve(v)) for k, v in cfg["paths"].items()}
    return cfg


def load_batches(cfg: dict) -> dict[str, list[str]]:
    b = yaml.safe_load(Path(cfg["paths"]["batch_manifest"]).read_text(encoding="utf-8"))
    return {k: [str(x) for x in v] for k, v in b.items()}


def check_batches(b: dict[str, list[str]]) -> None:
    """Patient-level leakage rules for a sequential manifest."""
    rounds = sorted({int(k.rsplit("_", 1)[1]) for k in b if k.startswith("future_eval_")})
    for k in rounds:
        fut = set(b[f"future_eval_{k}"])
        used = set()
        for j in range(1, k + 1):
            used |= set(b.get(f"correction_batch_{j}", [])) | set(b.get(f"validation_gate_{j}", []))
        if fut & used:
            raise ValueError(f"future_eval_{k} overlaps correction/validation patients of rounds <= {k}: {sorted(fut & used)[:5]}")
        if set(b.get(f"validation_gate_{k}", [])) & (set(b.get(f"correction_batch_{k}", [])) | fut):
            raise ValueError(f"validation_gate_{k} overlaps correction or future patients")
    if "holdout" in b:
        others = set().union(*[set(v) for k, v in b.items() if k != "holdout"])
        if set(b["holdout"]) & others:
            raise ValueError("holdout overlaps other batches")
