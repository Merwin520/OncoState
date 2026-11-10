"""Review items derived from the CURRENT patient state (no reference data). Used by the clinician UI.

  pair         predicted (a, b): "Are these the same lesion?"
  disappeared  baseline a left unmatched: "Has this lesion disappeared, or is it one of the follow-up lesions?"
  new          follow-up b left unmatched: "Is this a new lesion, or the continuation of a baseline lesion?"
"""
from __future__ import annotations

import numpy as np

from repair.matching_features import distance_matrix


def review_items(state: dict) -> list[dict]:
    items = []
    for c in state["correspondences"]:
        a, b = c["baseline_lesion_id"], c["followup_lesion_id"]
        if c["status"] == "persistent":
            items.append({"item_id": f"pair:{a}:{b}", "kind": "pair", "baseline_lesion_id": a, "followup_lesion_id": b,
                          "confidence": c.get("confidence"), "centroid_distance_mm": c.get("centroid_distance_mm"), "source": c["source"], "verified": bool(c.get("verified_by_event"))})
        elif c["status"] == "disappearing":
            items.append({"item_id": f"dis:{a}", "kind": "disappeared", "baseline_lesion_id": a, "followup_lesion_id": None,
                          "confidence": None, "centroid_distance_mm": None, "source": c["source"], "verified": bool(c.get("verified_by_event"))})
        else:
            items.append({"item_id": f"new:{b}", "kind": "new", "baseline_lesion_id": None, "followup_lesion_id": b,
                          "confidence": None, "centroid_distance_mm": None, "source": c["source"], "verified": bool(c.get("verified_by_event"))})
    order = {"pair": 0, "disappeared": 1, "new": 2}
    return sorted(items, key=lambda x: (order[x["kind"]], x["confidence"] if x["confidence"] is not None else 1.0, x["item_id"]))


def alternative_candidates(state: dict, baseline_lesion_id: str | None, k: int = 5, max_mm: float = 100.0) -> list[dict]:
    """Nearest follow-up lesions to a baseline lesion after registration (what the clinician can pick instead)."""
    case = {"patient_id": state["patient_id"], "baseline_lesions": state["baseline_lesions"], "followup_lesions": state["followup_lesions"]}
    bids, fids, d = distance_matrix(case)
    if baseline_lesion_id is None or baseline_lesion_id not in bids or not fids:
        return []
    row = d[bids.index(baseline_lesion_id)]
    order = np.argsort(row)
    fu = {l["lesion_id"]: l for l in state["followup_lesions"]}
    return [{"followup_lesion_id": fids[j], "distance_mm": float(row[j]), "diameter_mm": fu[fids[j]]["diameter_mm"],
             "volume_mm3": fu[fids[j]]["volume_mm3"]} for j in order[:k] if row[j] <= max_mm]


def nearest_baselines(state: dict, followup_lesion_id: str, k: int = 5, max_mm: float = 100.0) -> list[dict]:
    case = {"patient_id": state["patient_id"], "baseline_lesions": state["baseline_lesions"], "followup_lesions": state["followup_lesions"]}
    bids, fids, d = distance_matrix(case)
    if followup_lesion_id not in fids or not bids:
        return []
    col = d[:, fids.index(followup_lesion_id)]
    return [{"baseline_lesion_id": bids[i], "distance_mm": float(col[i])} for i in np.argsort(col)[:k] if col[i] <= max_mm]


def item_still_open(state: dict, item: dict) -> bool:
    """An item disappears from the state once a previous decision changed it (e.g. the pair was already re-assigned)."""
    return any(x["item_id"] == item["item_id"] for x in review_items(state))
