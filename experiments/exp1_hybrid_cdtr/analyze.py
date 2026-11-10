#!/usr/bin/env python3
"""Pre-specified endpoints: hybrid + CDT-R vs original V0 + CDT-R on the same 274 patients.
    python analyze.py  -> tables/*.csv, results/summary.json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REL = REPO / 'experiments/exp1_clinical_decision_reliability'
T = HERE / 'tables'; T.mkdir(exist_ok=True)
PROVED = 'PROVED_EQUIVALENT'
B, SEED = 10_000, 20261020
cp = lambda k, n: ((0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(.975, k + 1, n - k)))  # noqa: E731
pct = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))  # noqa: E731


def topk_error(score, wrong, k):
    if k <= 0:
        return np.nan
    o = np.argsort(-score, kind='stable'); s, w = score[o], wrong[o]; t = s[k - 1]
    above, tied = s > t, s == t
    return (w[above].sum() + (k - above.sum()) / tied.sum() * w[tied].sum()) / k


H = pd.read_csv(HERE / 'results/per_patient_hybrid.csv', dtype={'patient_id': str})
V = pd.read_csv(REPO / 'experiments/cdtr_exp1_state/results/per_patient_states.csv', dtype={'patient_id': str})
V = V[V.variant == 'crossfit']
pids = sorted(V.patient_id.unique()); N = len(pids)
rng = np.random.default_rng(SEED); IDX = rng.integers(0, N, (B, N))


def arrays(df, a):
    d = df[np.isclose(df.alpha, a)].set_index('patient_id').loc[pids]
    inc = (d.ref_included if 'ref_included' in d else d.graph_covered).to_numpy(bool)
    return d.status.eq(PROVED).to_numpy(), d.wrong.to_numpy(float), inc, d


summ = {}
# ------------------------------------------------------------------ 1-3: certification, certified error, coverage (all alphas, all repeats)
rows = []
for a in sorted(H.alpha.unique()):
    cV, wV, iV, _ = arrays(V, a)
    rows.append(dict(matcher='V0', repeat='-', alpha=a, n_cert=int(cV.sum()), cert_rate=cV.mean(), cert_wrong=int(wV[cV].sum()),
                     cert_err=wV[cV].mean() if cV.any() else np.nan, cov_all=iV.mean(), cov_cert=iV[cV].mean() if cV.any() else np.nan,
                     overall_err=wV.mean()))
    for r in sorted(H.repeat.unique()):
        cH, wH, iH, _ = arrays(H[H.repeat == r], a)
        rows.append(dict(matcher='hybrid', repeat=r, alpha=a, n_cert=int(cH.sum()), cert_rate=cH.mean(), cert_wrong=int(wH[cH].sum()),
                         cert_err=wH[cH].mean() if cH.any() else np.nan, cov_all=iH.mean(), cov_cert=iH[cH].mean() if cH.any() else np.nan,
                         overall_err=wH.mean()))
ALL = pd.DataFrame(rows); ALL.to_csv(T / 'certification_by_alpha.csv', index=False)

# ------------------------------------------------------------------ primary alpha 0.10: repeat 0 with CIs, paired vs V0
cV, wV, iV, dV = arrays(V, .10)
H0 = H[H.repeat == 0]
cH, wH, iH, dH = arrays(H0, .10)
prim = []
for name, c, w, i in (('V0 + CDT-R', cV, wV, iV), ('hybrid + CDT-R (repeat 0)', cH, wH, iH)):
    k, n = int(w[c].sum()), int(c.sum())
    prim.append(dict(arm=name, n_cert=n, cert_rate=n / N, cert_rate_ci=cp(n, N), cert_wrong=k, cert_err=k / n, cert_err_ci=cp(k, n),
                     cov_all=i.mean(), cov_all_ci=cp(int(i.sum()), N), cov_cert=i[c].mean(), cov_cert_ci=cp(int(i[c].sum()), n),
                     uncert_err=w[~c].mean(), overall_err=w.mean(), cert_wrong_ref_in=int(w[c & i].sum()), cert_wrong_ref_out=int(w[c & ~i].sum())))


def paired(ix):
    a_, b_ = cH[ix], cV[ix]
    e_h = wH[ix][a_].mean() if a_.any() else np.nan; e_v = wV[ix][b_].mean() if b_.any() else np.nan
    return dict(d_cert_rate=a_.mean() - b_.mean(), d_cert_err=e_h - e_v,
                d_cov_cert=(iH[ix][a_].mean() if a_.any() else np.nan) - (iV[ix][b_].mean() if b_.any() else np.nan),
                d_overall_err=wH[ix].mean() - wV[ix].mean())


pt = paired(np.arange(N)); bt = pd.DataFrame([paired(ix) for ix in IDX])
PD_ = pd.DataFrame([dict(endpoint=k, diff=v, lo=pct(bt[k])[0], hi=pct(bt[k])[1]) for k, v in pt.items()])
PD_.to_csv(T / 'paired_hybrid_minus_v0.csv', index=False)
pd.DataFrame(prim).to_csv(T / 'primary_alpha010.csv', index=False)

# ------------------------------------------------------------------ 5: decision- vs lesion-level selection at the hybrid-CDT-R coverage
conf = dH.hyb_min_conf.to_numpy(float); amb = -dH.ambiguous_lesions.to_numpy(float)


def sel(ix):
    c, w = cH[ix], wH[ix]; k = int(c.sum())
    e_c = w[c].mean() if k else np.nan
    e_conf, e_amb = topk_error(conf[ix], w, k), topk_error(amb[ix], w, k)
    return dict(cdtr=e_c, conf=e_conf, amb=e_amb, d_conf=e_c - e_conf, d_amb=e_c - e_amb, d_pool=e_c - (e_conf + e_amb) / 2)


sp = sel(np.arange(N)); sb = pd.DataFrame([sel(ix) for ix in IDX])
SEL = pd.DataFrame([dict(quantity=k, value=v, lo=pct(sb[k])[0], hi=pct(sb[k])[1]) for k, v in sp.items()])
SEL.to_csv(T / 'selection_hybrid.csv', index=False)

# ------------------------------------------------------------------ 6: fate of the nine incorrectly V0-certified patients
nine = dV[(dV.status == PROVED) & (dV.wrong == 1)].index.tolist()
fate = []
for r in sorted(H.repeat.unique()):
    _, _, _, d = arrays(H[H.repeat == r], .10)
    for p in nine:
        x = d.loc[p]
        st_ = 'certified correct' if (x.status == PROVED and not x.wrong) else ('certified wrong' if x.status == PROVED else 'not certified')
        fate.append(dict(repeat=r, patient_id=p, fate=st_, hybrid_wrong=int(x.wrong), ref_in_hybrid_set=int(x.ref_included)))
F_ = pd.DataFrame(fate); F_.to_csv(T / 'nine_v0_failures_under_hybrid.csv', index=False)
fate_ct = F_.groupby(['repeat', 'fate']).size().unstack(fill_value=0)
# the hybrid's own incorrect certificates
own = []
for r in sorted(H.repeat.unique()):
    _, _, _, d = arrays(H[H.repeat == r], .10)
    bad = d[(d.status == PROVED) & (d.wrong == 1)]
    for p, x in bad.iterrows():
        own.append(dict(repeat=r, patient_id=p, hyb_response=x.hyb_response, ref_response=x.ref_response, ref_included=int(x.ref_included),
                        in_v0_nine=int(p in nine), v0_status=dV.at[p, 'status'], v0_wrong=int(dV.at[p, 'wrong'])))
OWN = pd.DataFrame(own); OWN.to_csv(T / 'hybrid_incorrect_certificates.csv', index=False)

hp = H[np.isclose(H.alpha, .10)]
summ = dict(primary=prim, paired=PD_.to_dict('records'), selection=SEL.to_dict('records'),
            repeats_alpha010=ALL[(ALL.matcher == 'hybrid') & np.isclose(ALL.alpha, .10)][['repeat', 'n_cert', 'cert_wrong', 'cov_all', 'cov_cert']].to_dict('records'),
            nine_fate=fate_ct.to_dict('index'), argmin_eq_hybrid=float(hp.argmin_eq_hybrid.mean()),
            hybrid_point_error=float(hp.drop_duplicates(['repeat', 'patient_id']).groupby('repeat').wrong.mean().mean()),
            timeouts=int((H.status == 'INCONCLUSIVE_TIMEOUT').sum()))
(HERE / 'results/summary.json').write_text(json.dumps(summ, indent=1, default=float))
print(pd.DataFrame(prim).round(3).T.to_string()); print(PD_.round(3).to_string()); print(SEL.round(3).to_string())
print(ALL[np.isclose(ALL.alpha, .10)].round(3).to_string()); print(fate_ct); print(OWN.to_string())
print(ALL.pivot_table(index='alpha', columns=['matcher', 'repeat'], values='cert_wrong').to_string())
print('argmin==hybrid', summ['argmin_eq_hybrid'], 'hybrid point error', summ['hybrid_point_error'], 'timeouts', summ['timeouts'])
