#!/usr/bin/env python3
"""CDT-R driver.
  python run_cdtr.py calibrate            # conformal q from the calibration split ONLY -> results/calibration.json
  python run_cdtr.py run --split S        # CDT-R policy on split S (policy_validation | test) -> results/<S>/cdtr.json
Baselines: run_baselines.py. Metrics: evaluate.py. Report: make_report.py."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cdtr import schema as S  # noqa: E402
from cdtr.conformal import quantile  # noqa: E402
from cdtr.data import check_no_leakage, load_evidence  # noqa: E402
from cdtr.energy import best, prepare  # noqa: E402
from cdtr.reference import ReferenceForSplit, ReviewTool  # noqa: E402
from cdtr.agent import run_policy  # noqa: E402

RES = HERE / 'results'
CFG_PATH = HERE / 'configs/cdtr.yaml'


def cfg():
    return yaml.safe_load(CFG_PATH.read_text())


def calibrate():
    c = cfg(); check_no_leakage()
    ev = load_evidence('calibration'); ref = ReferenceForSplit('calibration')
    scores = {}
    for p, e in sorted(ev.items()):
        P = prepare(e, c['energy']['gate_mm'])
        _, F0 = best(P)
        scores[p] = P.F(ref.pairs(p)) - F0
    alphas = [c['conformal']['alpha_primary']] + c['conformal']['alpha_sensitivity']
    out = dict(n=len(scores), scores=scores, q={str(a): quantile(list(scores.values()), a) for a in alphas},
               config_sha256=hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(), created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (RES / 'calibration.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['q']))


def run_one(policy, split, alpha=None, seeds=(0,), tag=None):
    c = cfg(); check_no_leakage()
    cal = json.loads((RES / 'calibration.json').read_text())
    alpha = alpha if alpha is not None else c['conformal']['alpha_primary']
    q = cal['q'][str(alpha)]
    ev = load_evidence(split)
    n = len(ev)
    cps = sorted({int(round(r * n)) for r in c['review']['budgets_per_patient']})
    outdir = RES / split; outdir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in seeds:
        tool = ReviewTool(split, c['review']['cost_per_lesion'])
        res, meta, states = run_policy(policy, ev, q, c, tool, max(cps), cps, seed=seed)
        assert tool.calls == meta['reviews_used']
        runs[str(seed)] = dict(meta=meta, checkpoints={str(k): v for k, v in res.items()},
                               witnesses={p: dict(status=st.check.status, response=st.check.reference_response,
                                                  removed=st.check.witness_pairs_removed, added=st.check.witness_pairs_added,
                                                  lesions=st.check.witness_lesions, gap=st.check.counterexample_gap,
                                                  cex_response=st.check.counterexample.response if st.check.counterexample else None,
                                                  log=st.check.solver_log, seconds=st.check.solver_seconds)
                                          for p, st in states.items() if st.check is not None},
                               tool_log=[a.__dict__ for a in tool.log])
    blob = dict(policy=policy, split=split, alpha=alpha, q=q, checkpoints=cps, config_sha256=hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(),
                split_sha256=json.loads((RES / 'splits.json').read_text())['split_sha256'][split], runs=runs)
    name = tag or (policy if alpha == c['conformal']['alpha_primary'] else f'{policy}_alpha{alpha}')
    (outdir / f'{name}.json').write_text(json.dumps(blob, default=list))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['calibrate', 'run'])
    ap.add_argument('--split', default='policy_validation'); ap.add_argument('--alpha', type=float)
    a = ap.parse_args()
    if a.stage == 'calibrate':
        calibrate()
    else:
        run_one('cdtr', a.split, a.alpha)
