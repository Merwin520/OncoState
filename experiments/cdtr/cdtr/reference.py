"""The ONLY reader of results/reference_HIDDEN.json. Two entry points:
  ReferenceForSplit(split)   -> full reference for one split (calibration scores, evaluation); refuses the test split unless
                                `final_evaluation=True` is passed by evaluate.py
  ReviewTool(split)          -> SIMULATED review: answers one follow-up lesion's reference partner per call, counts every call.
Policies receive a ReviewTool, never a ReferenceForSplit."""
import json
from pathlib import Path

from . import schema as S

RES = Path(__file__).resolve().parents[1] / 'results'
SIMULATED = 'reference annotation (SIMULATED review; NOT clinician review)'


def _load(split):
    man = json.loads((RES / 'splits.json').read_text())
    ids = set(man['patients'][split])
    ref = json.loads((RES / 'reference_HIDDEN.json').read_text())
    return {p: dict(pairs=[tuple(x) for x in v['pairs']], baseline_class=v['baseline_class'], followup_class=v['followup_class'])
            for p, v in ref.items() if p in ids}


class ReferenceForSplit:
    def __init__(self, split, final_evaluation=False):
        if split == 'test' and not final_evaluation:
            raise PermissionError('test reference is only readable by the final evaluation')
        self._ref = _load(split)

    def pairs(self, pid):
        return self._ref[pid]['pairs']

    def __contains__(self, pid):
        return pid in self._ref


class ReviewTool:
    def __init__(self, split, cost=1.0):
        self._ref = _load(split)
        self.cost, self.calls, self.log = cost, 0, []

    def __call__(self, pid, followup_id, policy, step):
        partner = {f: b for b, f in self._ref[pid]['pairs']}.get(followup_id)
        self.calls += 1
        self.log.append(S.ToolAction(pid, followup_id, 'reference_identity_lookup', self.cost, policy, step))
        return S.EvidenceUpdate(pid, followup_id, partner, SIMULATED, step)
