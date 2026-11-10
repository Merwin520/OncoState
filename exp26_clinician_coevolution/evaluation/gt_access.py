"""The ONLY entry point to hidden reference (ground-truth) data.

Allowed callers: evaluation/, oracle/, repair/validation_gate.py, repair/non_regression.py and tests/.
The clinician app must never import this module (enforced by tests/test_gt_separation.py).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

ALLOWED_CALLER_PACKAGES = ("evaluation", "oracle", "repair", "tests", "figures", "scripts", "adapters")


def _check_caller() -> None:
    for frame in inspect.stack()[2:8]:
        mod = frame.frame.f_globals.get("__name__", "")
        if mod.startswith("app") or "/app/" in frame.filename.replace("\\", "/"):
            raise PermissionError("GT access from the clinician app is forbidden")


def load_gt(bundle_dir: str | Path) -> dict[str, dict]:
    """{patient_id: {'pairs': set[(a, b)], 'baseline_class': {...}, 'followup_class': {...}, 'change': {(a, b): cls}, 'burden': str}}"""
    _check_caller()
    d = Path(bundle_dir) / "gt_hidden"
    out = {}
    for p in sorted(d.glob("*.json")):
        g = json.loads(p.read_text())
        out[g["patient_id"]] = {
            "pairs": {tuple(x) for x in g["pairs"]},
            "baseline_class": g["baseline_class"], "followup_class": g["followup_class"],
            "change": {tuple(k.split("|")): v for k, v in g.get("change", {}).items()},
            "burden": g.get("burden"),
        }
    return out
