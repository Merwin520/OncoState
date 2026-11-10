#!/usr/bin/env python3
"""Supplementary analysis (descriptive): certification and error stratified by the
V0 rule-derived response category, and the evidence behind certified PD calls. Reads results/per_patient.csv + saved evidence/reference;
no model re-run.   -> tables/D_by_category.csv, tables/D_pd_evidence.csv, tables/D_coverage_vs_nominal.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; P1 = REPO / 'experiments/cdtr'
sys.path.insert(0, str(P1))
from cdtr.energy import best, prepare
cp = lambda k, n: ((0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(.975, k + 1, n - k))) if n else (np.nan, np.nan)
pp = pd.read_csv(HERE / 'results/per_patient.csv', dtype={'patient_id': str})
st = pd.read_csv(REPO / 'experiments/cdtr_exp1_state/results/per_patient_states.csv', dtype={'patient_id': str})
prim = st[(st.variant == 'crossfit') & np.isclose(st.alpha, .1)].set_index('patient_id')
pp['identity_certain'] = pp.patient_id.map(prim.identity_certain)
ev = json.loads((P1 / 'results/evidence.json').read_text())
ref = {p: [tuple(x) for x in v['pairs']] for p, v in json.loads((P1 / 'results/reference_HIDDEN.json').read_text()).items()}
rows = []
for p in pp.patient_id:
    P = prepare(ev[p], 30.0); x0, _ = best(P)
    _, i0, _ = P.response(x0); _, ir, _ = P.response(ref[p])
    rows.append(dict(patient_id=p, v0_new=int(i0['new_lesions']), ref_new=int(ir['new_lesions'])))
pp = pp.merge(pd.DataFrame(rows), on='patient_id')
# --- by V0 category
out = []
for cat in ['CR', 'PR', 'SD', 'PD', 'NE']:
    g = pp[pp.v0_response == cat]; c = g[g.certified == 1]; u = g[g.certified == 0]
    cu = c[c.identity_certain == 0]
    out.append(dict(v0_category=cat, n=len(g), certified=len(c), cert_rate=len(c) / len(g) if len(g) else np.nan,
                    certified_identity_unresolved=len(cu), cert_wrong=int(c.wrong.sum()),
                    cert_err=c.wrong.mean() if len(c) else np.nan, cert_err_lo=cp(int(c.wrong.sum()), len(c))[0], cert_err_hi=cp(int(c.wrong.sum()), len(c))[1],
                    uncert_err=u.wrong.mean() if len(u) else np.nan, uncert_wrong=int(u.wrong.sum()), uncert_n=len(u)))
D = pd.DataFrame(out); D.to_csv(HERE / 'tables/D_by_category.csv', index=False); print(D.round(3).to_string())
# --- certified PD: driver and reference support
pd_c = pp[(pp.certified == 1) & (pp.v0_response == 'PD')]
def cls(r):
    if not r.v0_new: return 'sum-increase PD'
    if r.ref_new: return 'new lesion, also new in reference'
    return 'new lesion, matched in reference'
pd_c = pd_c.assign(evidence=pd_c.apply(cls, axis=1))
E = pd_c.groupby(['evidence', 'identity_certain']).agg(n=('wrong', 'size'), wrong=('wrong', 'sum')).reset_index()
E.to_csv(HERE / 'tables/D_pd_evidence.csv', index=False); print(E.to_string())
print('certified PD total', len(pd_c), 'ref PD', int((pd_c.ref_response == 'PD').sum()))
# non-PD certified with unresolved identity
npd = pp[(pp.certified == 1) & (pp.v0_response != 'PD')]
print('certified non-PD', len(npd), 'unresolved', int((npd.identity_certain == 0).sum()), 'wrong', int(npd.wrong.sum()))
# --- coverage vs nominal
C = []
for (var, a), g in st.groupby(['variant', 'alpha']):
    c = g.status.eq('PROVED_EQUIVALENT')
    C.append(dict(variant=var, alpha=a, nominal=1 - a, coverage_all=g.graph_covered.mean(), coverage_certified=g[c].graph_covered.mean() if c.any() else np.nan,
                  n_certified=int(c.sum())))
pd.DataFrame(C).to_csv(HERE / 'tables/D_coverage_vs_nominal.csv', index=False); print(pd.DataFrame(C).round(3).to_string())
