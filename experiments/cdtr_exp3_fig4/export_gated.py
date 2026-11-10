#!/usr/bin/env python3
"""Rebuild the gated updated matcher from the reader-study consensus (correspondence chosen by >= 2 of 3 readers; events as in
reader_study/model_update.py), fit_candidate,
run_gate on the 80 gate patients) and export its energy for CDT-R: cost_ij = -logit(p_ij) on candidate pairs (p clipped to
[1e-6, 1-1e-6]; non-candidate pairs get p = 1e-6), 'unmatched' = 0. The argmin of this energy equals the gated matcher (checked)."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PKG = REPO / 'exp26_clinician_coevolution'
os.chdir(PKG); sys.path.insert(0, str(PKG)); sys.path.insert(0, str(REPO / 'experiments/cdtr'))
from adapters.schema_mapping import load_bundle  # noqa: E402
from configs.loader import load_config  # noqa: E402
from evaluation.clinician_coevolution import SOURCE, make_batches  # noqa: E402
from evaluation.cohorts import ALL_REVIEW_PATIENTS  # noqa: E402
from evaluation.gt_access import load_gt  # noqa: E402
from repair.build_candidate import fit_candidate  # noqa: E402
from repair.matching_features import candidate_pairs  # noqa: E402
from repair.matching_repair import CentroidHungarianMatcher  # noqa: E402
from repair.validation_gate import run_gate  # noqa: E402
from cdtr.energy import Prepared, best  # noqa: E402
from cdtr.rules import tenths  # noqa: E402

cfg = load_config('configs/experiment_real.yaml'); seed = cfg['random_seed']
bundle = load_bundle(SOURCE); cases, gt = bundle['cases'], load_gt(SOURCE)
bt = make_batches(bundle['manifest'], json.loads(ALL_REVIEW_PATIENTS.read_text()))
sys.path.insert(0, str(REPO / 'experiments/reader_study'))
from model_update import OUT as READERS, events_of  # noqa: E402
import pandas as pd  # noqa: E402
rows = pd.read_csv(READERS / 'readers3_rows.csv', dtype=str, keep_default_na=False)
src = 'clinician (2 of 3 consensus)'
cand = fit_candidate(events_of(rows, cases, 'consensus'), cases, cfg, seed)
v0 = CentroidHungarianMatcher(cfg['v0']['gate_mm'])
g = run_gate(cases, gt, v0, cand, bt['validation_gate_1'], cfg['non_regression_thresholds'], cfg['bootstrap_iterations'], seed)
print('event source', src, 'gate accepted', g['accepted'])
assert g['accepted']
out, agree = {}, 0
for split, pids in (('calibration', bt['validation_gate_1']), ('test', bt['future_eval_1'])):
    for p in pids:
        c = cases[p]
        bids = [l['lesion_id'] for l in c['baseline_lesions']]; fids = [l['lesion_id'] for l in c['followup_lesions']]
        prob = np.full((len(bids), len(fids)), 1e-6)
        cp = candidate_pairs(c, set(v0.predict(c)['pairs']), cand.max_distance_mm)
        if not cp.empty:
            pr = np.clip(cand.proba(cp), 1e-6, 1 - 1e-6)
            for a, b, x in zip(cp.baseline_lesion_id, cp.followup_lesion_id, pr):
                prob[bids.index(a), fids.index(b)] = x
        cost = -np.log(prob / (1 - prob))
        gp = sorted(tuple(x) for x in cand.predict(c)['pairs'])
        P = Prepared(p, bids, fids, cost, 0.0, {l['lesion_id']: tenths(l['diameter_mm']) for l in c['baseline_lesions']},
                     {l['lesion_id']: tenths(l['diameter_mm']) for l in c['followup_lesions']})
        x, _ = best(P)
        agree += sorted(x) == gp
        if sorted(x) != gp:
            print('argmin != gated', p, P.F(x), P.F(gp))
        out[p] = dict(patient_id=p, split=split, bids=bids, fids=fids, cost=cost.tolist(), prob=prob.tolist(), gated_pairs=gp,
                      v0_pairs=sorted(tuple(x) for x in v0.predict(c)['pairs']), bl_t=P.bl_t, fu_t=P.fu_t)
print('argmin == gated matcher:', agree, '/', len(out))
(HERE / 'results/evidence_gated.json').write_text(json.dumps(dict(event_source=src, gate=dict(accepted=g['accepted'], reason=g['reason']),
                                                                   patients=out)))
