"""Adapter: Exp 2.6 longitudinal-CT bundle (Exp 2.2 setting-A lesions, registered centroids) -> CDT-R evidence + split manifest, and a
SEPARATE reference file. Policy code never reads the reference file; only `evaluate` and the simulated-review tool do.
  python -m cdtr.data   (run once; writes results/evidence.json, results/reference_HIDDEN.json, results/splits.json)"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
E26 = REPO / 'exp26_clinician_coevolution'
RES = ROOT / 'results'
SPLIT_OF_BATCH = {'validation_gate_1': 'calibration', 'development': 'policy_validation', 'correction_batch_1': 'policy_validation',
                  'future_eval_1': 'test'}


def export():
    sys.path.insert(0, str(E26))
    from adapters.schema_mapping import load_bundle
    from evaluation.clinician_coevolution import SOURCE, make_batches
    from evaluation.cohorts import ALL_REVIEW_PATIENTS
    from evaluation.gt_access import load_gt
    from repair.matching_features import registered_centroids
    from repair.matching_repair import CentroidHungarianMatcher
    v0m = CentroidHungarianMatcher(30.0)   # repository MatchingNode replica (Exp 2.6 V0)
    b = load_bundle(SOURCE); cases, gt = b['cases'], load_gt(SOURCE)
    batches = make_batches(b['manifest'], json.loads(ALL_REVIEW_PATIENTS.read_text()))
    split = {p: SPLIT_OF_BATCH[k] for k, ps in batches.items() for p in ps}
    assert len(split) == sum(len(v) for v in batches.values()), 'patient in more than one batch'
    ev, ref = {}, {}
    for p, sp in split.items():
        c = cases[p]
        bids, bc, fids, fc = registered_centroids(c)
        cb = dict(zip(bids, bc.tolist())); cf = dict(zip(fids, fc.tolist()))
        ev[p] = dict(patient_id=p, split=sp, v0_pairs_repository=sorted(list(x) for x in v0m.predict(c)['pairs']),
                     registration_residual_mm=c.get('registration', {}).get('quality', {}).get('residual_mm'),
                     baseline=[dict(lesion_id=l['lesion_id'], diameter_mm=float(l['diameter_mm']), volume_mm3=float(l['volume_mm3']),
                                    centroid_mm=cb[l['lesion_id']], provenance=l.get('mask_ref')) for l in c['baseline_lesions']],
                     followup=[dict(lesion_id=l['lesion_id'], diameter_mm=float(l['diameter_mm']), volume_mm3=float(l['volume_mm3']),
                                    centroid_mm=cf[l['lesion_id']], provenance=l.get('mask_ref')) for l in c['followup_lesions']])
        ref[p] = dict(pairs=sorted(list(x) for x in gt[p]['pairs']), baseline_class=gt[p]['baseline_class'], followup_class=gt[p]['followup_class'])
    RES.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(ev, sort_keys=True); (RES / 'evidence.json').write_text(blob)
    rblob = json.dumps(ref, sort_keys=True); (RES / 'reference_HIDDEN.json').write_text(rblob)
    man = dict(source=str(SOURCE), batches_to_splits=SPLIT_OF_BATCH, n={s: sum(v == s for v in split.values()) for s in set(split.values())},
               patients={s: sorted(p for p, v in split.items() if v == s) for s in set(split.values())},
               evidence_sha256=hashlib.sha256(blob.encode()).hexdigest(), reference_sha256=hashlib.sha256(rblob.encode()).hexdigest())
    for s in man['patients']:
        man.setdefault('split_sha256', {})[s] = hashlib.sha256(','.join(man['patients'][s]).encode()).hexdigest()
    (RES / 'splits.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(man['n']), man['evidence_sha256'][:12])


def load_evidence(split=None):
    ev = json.loads((RES / 'evidence.json').read_text())
    return {p: e for p, e in ev.items() if split is None or e['split'] == split}


def check_no_leakage():
    man = json.loads((RES / 'splits.json').read_text())
    sets = {k: set(v) for k, v in man['patients'].items()}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise AssertionError(f'patients shared between {a} and {b}')
    return {k: len(v) for k, v in sets.items()}


if __name__ == '__main__':
    export(); print(check_no_leakage())
