#!/usr/bin/env python3
"""CDT-R on top of the gated updated matcher, 144 unseen patients, retrospective reference-lookup evidence queries."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[0] / 'cdtr'
sys.path.insert(0, str(P1))
from cdtr.agent import run_policy  # noqa: E402
from cdtr.conformal import quantile  # noqa: E402
from cdtr.energy import Prepared, best  # noqa: E402
from cdtr.reference import ReferenceForSplit, ReviewTool  # noqa: E402

CFG = yaml.safe_load((HERE / 'configs/fig4.yaml').read_text())
EV = json.loads((HERE / 'results/evidence_gated.json').read_text())['patients']


def prep(e):
    P = Prepared(e['patient_id'], e['bids'], e['fids'], np.array(e['cost'], float).reshape(len(e['bids']), len(e['fids'])), 0.0,
                 e['bl_t'], e['fu_t'], [tuple(x) for x in e['gated_pairs']])
    P.prob = np.array(e['prob'], float).reshape(len(e['bids']), len(e['fids']))
    return P


def main(only=None):
    cal = {p: e for p, e in EV.items() if e['split'] == 'calibration'}
    test = {p: dict(e, patient_id=p) for p, e in EV.items() if e['split'] == 'test'}
    ref = ReferenceForSplit('calibration')
    sc = []
    for p, e in cal.items():
        P = prep(dict(e, patient_id=p)); _, F0 = best(P); sc.append(P.F(ref.pairs(p)) - F0)
    q = quantile(sc, CFG['conformal']['alpha_primary'])
    print('q gated energy', q, 'cal scores quantiles', np.quantile(sc, [.5, .8, .9]).round(2))
    n = len(test); cps = sorted({int(round(r * n)) for r in CFG['review']['budgets_per_patient']})
    (HERE / 'results/test').mkdir(parents=True, exist_ok=True)
    for pol in CFG['policies']:
        if only and pol != only:
            continue
        seeds = CFG['random_seeds'] if pol == 'random' else [0]
        runs = {}
        for s in seeds:
            tool = ReviewTool('test')
            res, meta, states = run_policy('v0_no_review' if pol == 'no_review' else pol, test, q, CFG, tool, max(cps), cps, seed=s, prep_fn=prep)
            runs[str(s)] = dict(meta=meta, checkpoints={str(k): v for k, v in res.items()}, tool_log=[a.__dict__ for a in tool.log])
        (HERE / f'results/test/gated_{pol}.json').write_text(json.dumps(dict(policy=pol, energy='gated', q=q, checkpoints=cps, runs=runs), default=list))
    if not only:
        (HERE / 'results/calibration.json').write_text(json.dumps(dict(q=q, scores=sc)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
