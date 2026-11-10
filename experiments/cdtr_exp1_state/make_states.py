#!/usr/bin/env python3
"""Illustrative members of the calibrated candidate set (patient ef0d3930a7, cross-fitted alpha = 0.10, as in Experiment 1).

Gumbel-Sinkhorn samples (tau = 15 mm, the Experiment 1 value) are drawn and kept when their energy lies inside the patient's
calibrated threshold q (Cα); each kept state is scored with the same deterministic RECIST rules. The CDT-R MILP verifier in
Experiment 1 already proved that every state in Cα gives the same response (status PROVED_EQUIVALENT); the samples only
illustrate members of that set.
    python make_states.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PID, N_SAMPLES, SEED = 'ef0d3930a7', 5000, 7
REPO = Path(__file__).resolve().parents[2]
P1 = REPO / 'experiments/cdtr'
EXP1 = REPO / 'experiments/cdtr_exp1_state'
sys.path.insert(0, str(P1)); sys.path.insert(0, str(REPO / 'exp26_clinician_coevolution'))
from cdtr.energy import best, prepare  # noqa: E402
from cdtr.sinkhorn import gumbel_samples  # noqa: E402

SHOW = [   # the six states drawn in the figure (all are members of Cα; listed by follow-up partner of each baseline lesion)
    {'B9': 'F4', 'B4': 'F3', 'B6': 'F5', 'B5': 'F1', 'B1': 'F2'},
    {'B9': 'F3', 'B4': 'F4', 'B6': 'F5', 'B5': 'F1', 'B1': 'F2'},
    {'B9': 'F4', 'B4': 'F3', 'B6': 'F5', 'B5': 'F2', 'B1': 'F1'},
    {'B9': 'F4', 'B4': 'F1', 'B6': 'F5', 'B5': 'F3', 'B1': 'F2'},
    {'B4': 'F3', 'B6': 'F5', 'B5': 'F1', 'B1': 'F2', 'B2': 'F4'},
    {'B9': 'F4', 'B4': 'F3', 'B5': 'F1', 'B1': 'F2', 'B2': 'F5'},
]


def main():
    ev = json.loads((P1 / 'results/evidence.json').read_text())
    P = prepare(ev[PID], 30.0); x0, F0 = best(P)
    tau = json.loads((EXP1 / 'results/meta.json').read_text())['tau']
    st = pd.read_csv(EXP1 / 'results/per_patient_states.csv', dtype={'patient_id': str})
    r = st[(st.patient_id == PID) & (st.variant == 'crossfit') & np.isclose(st.alpha, .10)].iloc[0]
    q = float(r.q)
    seen = {}
    for s in gumbel_samples(P, tau, N_SAMPLES, np.random.default_rng(SEED)):
        k = tuple(sorted(s))
        if k not in seen and P.F(s) - F0 <= q + 1e-9:
            resp, inp, _ = P.response(s); seen[k] = (resp, inp['s1_tenths'] / 10)
    shown = []
    for m in SHOW:
        x = tuple(sorted(m.items()))
        dF = P.F(list(x)) - F0
        assert dF <= q + 1e-9, (m, dF, q)
        resp, inp, _ = P.response(list(x))
        shown.append(dict(pairs=m, response=resp, fu_sum=inp['s1_tenths'] / 10, bl_sum=inp['s0_tenths'] / 10,
                          change=(inp['s1_tenths'] - inp['s0_tenths']) / inp['s0_tenths'], new_lesions=inp['new_lesions'],
                          energy_above_best=round(dF, 2)))
    sums = [v[1] for v in seen.values()]
    out = dict(patient_id=PID, alpha=.10, q=q, tau=tau, status=r.status, verifier_reachable=r.reachable,
               targets=P.response(x0)[1]['targets'], v0_pairs=[list(p) for p in x0],
               sampled_states_in_set=len(seen), sampled_responses=sorted({v[0] for v in seen.values()}),
               sampled_fu_sum_range=[min(sums), max(sums)], n_bl=P.n, n_fu=P.m, ambiguous_lesions=int(r.ambiguous_lesions),
               registration_residual_mm=float(r.residual), shown=shown)
    (Path(__file__).resolve().parent / 'results/example_candidate_states.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'shown'}, indent=1))
    for s in shown:
        print(s)


if __name__ == '__main__':
    main()
