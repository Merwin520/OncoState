#!/usr/bin/env python3
"""Case facts (patient 31b7b72368) from the repository engine and saved tables -> results/case_facts.json.
    python dialogue_facts.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'experiments/decision_validity'))
from dv.common import evidence, prepare  # noqa: E402

PID = '31b7b72368'
P = prepare(evidence()[PID], 30.0)
(r0, d0, _), (r1, d1, _) = P.response(sorted(P.v0_repository)), P.response([('B1', 'F1')])
acc = pd.read_csv(REPO / 'experiments/decision_validity/results/acceptance_per_patient.csv', dtype={'patient_id': str})
cd = acc[(acc.patient_id == PID) & (acc.method == 'C0') & np.isclose(acc.alpha, .1)].iloc[0]
rd = pd.read_csv(HERE / 'results/readers3_patients.csv', dtype={'patient_id': str}).set_index('patient_id').loc[PID]
s0, s1 = d1['s0_tenths'] / 10, d1['current_sum']
out = dict(patient=PID, r0=r0, r1=r1, s0=s0, s1=s1, pct=f'{100 * (s1 - s0) / s0:+.0f}%'.replace('-', '−'), status=cd.status,
           reachable=cd.reachable.split(';'), ref=rd.resp_ref, readers=[rd.resp_R02, rd.resp_R03, rd.resp_R04],
           diam={'B1': P.bl_t['B1'] / 10, 'F1': P.fu_t['F1'] / 10}, v0_new_lesion=bool(d0['new_lesions']), v1_new_lesion=bool(d1['new_lesions']))
out['n_targets'] = len(d1['targets'])
# decision-invariant comparison case: same kind of revision (V0 left B1 unmatched, F1 called new; both readers link B1 -> F1)
Q = prepare(evidence()['a597e50502'], 30.0)
(q0, e0, _), (q1, e1, _) = Q.response(sorted(Q.v0_repository)), Q.response(sorted(Q.v0_repository) + [('B1', 'F1')])
ci = acc[(acc.patient_id == 'a597e50502') & (acc.method == 'C0') & np.isclose(acc.alpha, .1)].iloc[0]
ri = pd.read_csv(HERE / 'results/readers3_patients.csv', dtype={'patient_id': str}).set_index('patient_id').loc['a597e50502']
out['inv'] = dict(patient='a597e50502', r0=q0, r1=q1, s0=e1['s0_tenths'] / 10, s1=e1['current_sum'],
                  pct=f"{100 * (e1['current_sum'] - e1['s0_tenths'] / 10) / (e1['s0_tenths'] / 10):+.0f}%".replace('-', '−'), status=ci.status,
                  reachable=ci.reachable.split(';'), ref=ri.resp_ref, readers=[ri.resp_R02, ri.resp_R03, ri.resp_R04], n_targets=len(e1['targets']))
# coverage-aware CDT-R, alpha 0.10: status and whether the reference state lies inside the calibrated set
for key, pid in (('ca', PID), ('ca_inv', 'a597e50502')):
    r = acc[(acc.patient_id == pid) & (acc.method == 'CA') & np.isclose(acc.alpha, .1)].iloc[0]
    r0_ = acc[(acc.patient_id == pid) & (acc.method == 'C0') & np.isclose(acc.alpha, .1)].iloc[0]
    out[key] = dict(status=r.status, reachable=r.reachable.split(';'), ref_in_set=bool(r.ref_score <= r.q), q=float(r.q), ref_score=float(r.ref_score),
                    c0_status=r0_.status, c0_ref_in_set=bool(r0_.ref_score <= r0_.q))
out['decisive'] = dict(case1=['new lesion', FACT_PCT := out['pct']], case2=['new lesion', 'target ' + out['inv']['pct']])
out['non_target'] = [d0['non_target'], d1['non_target']]
(HERE / 'results/case_facts.json').write_text(json.dumps(out, indent=1))
print(out)
