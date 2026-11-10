"""Deterministic RECIST 1.1 helper for research workflows.

Scope: target-lesion sum-of-diameters plus explicit non-target/new-lesion status.
This is deliberately not an image model and does not infer any required input.
It is a transparent rule engine for already-verified measurements.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import math


_ALLOWED_NONTARGET = {"CR", "NON-CR/NON-PD", "PD", "NE", "ABSENT"}


class RecistService:
    def evaluate(
        self,
        baseline_sum_mm: float,
        current_sum_mm: float,
        nadir_sum_mm: Optional[float] = None,
        all_target_lesions_disappeared: bool = False,
        all_pathologic_nodes_lt10mm: bool = False,
        non_target_response: str = "NE",
        new_lesions: Optional[bool] = None,
    ) -> Dict[str, Any]:
        baseline = float(baseline_sum_mm)
        current = float(current_sum_mm)
        if not math.isfinite(baseline) or not math.isfinite(current):
            raise ValueError("diameter sums must be finite")
        if baseline < 0 or current < 0:
            raise ValueError("diameter sums must be non-negative")
        if baseline == 0 and not all_target_lesions_disappeared:
            raise ValueError("baseline_sum_mm must be > 0 when measurable target lesions are present")

        non_target = str(non_target_response or "NE").strip().upper()
        if non_target not in _ALLOWED_NONTARGET:
            raise ValueError(f"non_target_response must be one of {sorted(_ALLOWED_NONTARGET)}")

        nadir = baseline if nadir_sum_mm is None else float(nadir_sum_mm)
        if not math.isfinite(nadir) or nadir > baseline:
            raise ValueError("nadir must be finite and no greater than baseline")
        if nadir < 0:
            raise ValueError("nadir_sum_mm must be non-negative")

        change_baseline_pct = None if baseline == 0 else (current - baseline) / baseline * 100.0
        change_nadir_pct = None if nadir == 0 else (current - nadir) / nadir * 100.0
        absolute_from_nadir = current - nadir

        # Target lesion response.
        if all_target_lesions_disappeared and all_pathologic_nodes_lt10mm:
            target = "CR"
            target_rule = "All target lesions disappeared and any pathologic lymph nodes are <10 mm short axis."
        elif nadir > 0 and current >= 1.20 * nadir and absolute_from_nadir >= 5.0:
            target = "PD"
            target_rule = "At least 20% increase from nadir with at least 5 mm absolute increase. PD overrides shrinkage from baseline."
        elif nadir == 0 and current > 0:
            target = "NE"
            target_rule = "Reappearance after a zero nadir requires lesion-level clinician review."
        elif baseline > 0 and current <= 0.70 * baseline:
            target = "PR"
            target_rule = "At least 30% decrease in target-lesion sum relative to baseline without PD from nadir."
        else:
            target = "SD"
            target_rule = "Neither PR nor PD criteria are met."

        # Overall response for the common setting with measurable target disease.
        if new_lesions:
            overall = "PD"
            overall_rule = "New lesion(s) explicitly present."
        elif non_target == "PD":
            overall = "PD"
            overall_rule = "Unequivocal progression of non-target disease explicitly recorded."
        elif target == "PD":
            overall = "PD"
            overall_rule = "Target-lesion PD criteria met."
        elif non_target == "NE" or new_lesions is None or target == "NE":
            overall = "NE"
            overall_rule = "Required non-target/new-lesion evidence is unknown or target assessment requires review."
        elif target == "CR" and non_target in {"CR", "ABSENT"}:
            overall = "CR"
            overall_rule = "Target and non-target disease both meet CR with no new lesions."
        elif target in {"CR", "PR"} and non_target in {"CR", "ABSENT", "NON-CR/NON-PD"}:
            overall = "PR"
            overall_rule = "Target response is CR/PR, non-target disease is not PD, and no new lesions are present."
        elif target == "SD" and non_target in {"CR", "ABSENT", "NON-CR/NON-PD"}:
            overall = "SD"
            overall_rule = "Target disease is SD, non-target disease is not PD, and no new lesions are present."
        else:
            overall = "NE"
            overall_rule = "Provided evidence does not support a complete overall response category in this helper's scope."

        return {
            "framework": "RECIST 1.1",
            "scope": "Research rule engine for verified measurements; does not infer lesions or substitute for clinician review.",
            "inputs": {
                "baseline_sum_mm": baseline,
                "current_sum_mm": current,
                "nadir_sum_mm": nadir,
                "all_target_lesions_disappeared": bool(all_target_lesions_disappeared),
                "all_pathologic_nodes_lt10mm": bool(all_pathologic_nodes_lt10mm),
                "non_target_response": non_target,
                "new_lesions": new_lesions,
            },
            "derived": {
                "change_from_baseline_pct": change_baseline_pct,
                "change_from_nadir_pct": change_nadir_pct,
                "absolute_change_from_nadir_mm": absolute_from_nadir,
            },
            "target_response": target,
            "target_rule": target_rule,
            "overall_response": overall,
            "overall_rule": overall_rule,
        }


_instance: Optional[RecistService] = None


def get_recist_service() -> RecistService:
    global _instance
    if _instance is None:
        _instance = RecistService()
    return _instance
