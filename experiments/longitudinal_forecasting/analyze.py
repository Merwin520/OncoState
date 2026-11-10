#!/usr/bin/env python3
"""longitudinal-forecasting analysis (pre-specified analysis plan): audit, B/H/S/M forecasts per landmark, missing-visit robustness, revision mechanism,
I-SPY1 external test. Models are fitted once per (representation, landmark, fold) on natural-scenario training patients and applied to
the test patients under every scenario.   python analyze.py [--jobs 48] [--no-memory]"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import OUT, SCEN, T2, ISPY1, ISPY2, F, table_from_memory

warnings.filterwarnings('ignore', category=ConvergenceWarning)
NB, SEED = 2000, 20261011
REPS = ['B', 'H', 'S', 'M', 'Htab', 'SA']
LD = ('ld',)


def fit(X, y):
    pipe = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                         LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=.5, max_iter=20000, random_state=0))
    return GridSearchCV(pipe, {'logisticregression__C': [.01, .03, .1, .3, 1]}, cv=3, scoring='roc_auc').fit(X, y)


def metrics(y, p):
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    cal = LogisticRegression(C=1e6).fit(lp[:, None], y)
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                cal_slope=float(cal.coef_[0, 0]), cal_intercept=float(cal.intercept_[0]))


def load_memory(cohort, scen):
    f = OUT / 'raw' / f'memory_{cohort}_{scen}.jsonl'
    if not f.exists():
        return {}
    out = {}
    for l in f.read_text().splitlines():
        j = json.loads(l); out[j['patient_id']] = j
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=48); ap.add_argument('--no-memory', action='store_true')
    ap.add_argument('--run-name', default='analysis', help='output folder under outputs/longitudinal_forecasting')
    a = ap.parse_args()
    run = OUT / a.run_name; run.mkdir(parents=True, exist_ok=True)
    I = ISPY2(); pts = I.pts
    y = np.array([int(I.clin.at[p, 'pCR']) for p in pts])
    sub = [I.src.clinical(p)['fields'].get('clinical_subtype') for p in pts]
    clin = {p: I.clinical(p) for p in pts}
    arms = sorted({c['arm'] for c in clin.values()})
    # ------------------------------------------------------------------ audit
    obs = {p: {tp: I.record(p, tp, 'natural', 3) is not None for tp in T2} for p in pts}
    E = ISPY1()
    aud = [dict(cohort='I-SPY2', item='synchronized patients', n=len(pts)),
           dict(cohort='I-SPY2', item='with pCR outcome (evaluated)', n=int(len(y))),
           dict(cohort='I-SPY2', item='pCR prevalence', n=float(y.mean())),
           dict(cohort='I-SPY2', item='with T0 and T1 FTV', n=sum(obs[p]['T0'] and obs[p]['T1'] for p in pts)),
           dict(cohort='I-SPY2', item='patients with >=1 visit without MRI', n=sum(not all(obs[p].values()) for p in pts))]
    aud += [dict(cohort='I-SPY2', item=f'MRI available at {tp}', n=sum(obs[p][tp] for p in pts)) for tp in T2]
    aud += [dict(cohort='I-SPY2', item=f'evaluated at landmark {tp}', n=len(pts)) for tp in T2]
    aud += [dict(cohort='I-SPY1', item='patients in TCIA subset', n=len(E.all)),
            dict(cohort='I-SPY1', item='pCR missing (excluded)', n=int(E.src.data.PCR.isna().sum())),
            dict(cohort='I-SPY1', item='baseline LD missing among those with pCR (excluded)',
                 n=int((E.src.data.PCR.notna() & E.src.data['MRI LD Baseline'].isna()).sum())),
            dict(cohort='I-SPY1', item='external test patients', n=len(E.pts)),
            dict(cohort='I-SPY1', item='pCR prevalence', n=float(np.mean([E.y(p) for p in E.pts])))]
    aud += [dict(cohort='I-SPY1', item=f'MRI LD available at {c}', n=int(E.src.data.loc[E.pts, c].notna().sum()))
            for c in ('MRI LD Baseline', 'MRI LD 1-3dAC', 'MRI LD InterReg', 'MRI LD PreSurg')]
    pd.DataFrame(aud).to_csv(run / 'cohort_audit.csv', index=False)
    print(pd.DataFrame(aud).to_string(index=False), flush=True)

    # ------------------------------------------------------------------ feature tables
    mem = {s: ({} if a.no_memory else load_memory('ispy2', s)) for s in SCEN}
    st = {s: {p: I.states(p, s) for p in pts} for s in SCEN}
    nd = {p: I.states(p, 'revision', durable=False) for p in pts}

    def feats(rep, scen, k, p):
        if rep == 'B':
            return F(I.table_from_source(p, scen, 0), 0, clin[p])
        if rep == 'H':
            return F(I.table_from_source(p, scen, k), k, clin[p])
        if rep == 'Htab':
            return F(I.table_from_source(p, scen, k), k, clin[p], with_flags=False, summaries=False)
        if rep in ('S', 'SA'):
            f = F(I.table_from_state(st[scen][p][k], k), k, clin[p])
            if rep == 'SA':
                f.update({f'arm_{x}': float(clin[p]['arm'] == x) for x in arms})
            return f
        if rep == 'ND':
            return F(I.table_from_state(nd[p][k], k), k, clin[p])
        if rep == 'M':
            m = mem[scen].get(p)
            return F(table_from_memory(m['snapshots'][k] if m else None, k, ('ftv_pe', 'ftv_ser', 'ld', 'sph', 'bpe')), k, clin[p])
    reps = [r for r in REPS if not (r == 'M' and a.no_memory)]
    X = {}
    for scen in SCEN:
        for rep in reps + (['ND'] if scen == 'revision' else []):
            if scen != 'natural' and rep in ('B', 'Htab', 'SA'):
                continue
            for k in range(4):
                if rep == 'M' and not mem[scen]:
                    continue
                rows = [feats(rep, scen, k, p) for p in pts]
                X[(rep, scen, k)] = pd.DataFrame(rows).to_numpy(float), list(rows[0].keys())
    # memory fidelity vs source (natural): values transcribed correctly
    if not a.no_memory and mem['natural']:
        fid = []
        for p in pts:
            ms = table_from_memory(mem['natural'][p]['snapshots'][3], 3, ('ftv_pe', 'ftv_ser', 'ld', 'sph', 'bpe'))
            src = I.table_from_source(p, 'natural', 3)
            for tp in T2:
                for n in ('ftv_pe', 'ftv_ser', 'ld', 'sph', 'bpe'):
                    a_, b_ = (src.get(tp) or {}).get(n), (ms.get(tp) or {}).get(n)
                    fid.append(dict(patient_id=p, visit=tp, field=n, source=a_, memory=b_,
                                    correct=(a_ is None and b_ is None) or (a_ is not None and b_ is not None and abs(a_ - b_) <= 1e-6 * max(1, abs(a_)))))
        fid = pd.DataFrame(fid); fid.to_csv(run / 'memory_fidelity.csv', index=False)
        fails = sum(not g['ok'] for j in mem['natural'].values() for g in j['logs'])
        print('memory transcription fidelity', fid.correct.mean(), 'patients with any error', fid.groupby('patient_id').correct.all().eq(False).sum(),
              'failed updates', fails, flush=True)

    # ------------------------------------------------------------------ CV
    strata = np.array([f'{a}{b}' for a, b in zip(y, sub)])
    folds = [(r, i, tr, te) for r in range(5) for i, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(pts, strata))]
    pd.DataFrame([dict(repeat=r, fold=i, patient_id=pts[j]) for r, i, tr, te in folds for j in te]).to_csv(run / 'folds.csv', index=False)

    def job(rep, k, r, i, tr, te):
        Xn, _ = X[(rep, 'natural', k)]
        m = fit(Xn[tr], y[tr])
        out = {('natural', rep): m.predict_proba(Xn[te])[:, 1]}
        for scen in SCEN:
            if scen == 'natural':
                continue
            for rr in ([rep] + (['ND'] if rep == 'S' and scen == 'revision' else [])):
                if (rr, scen, k) in X:
                    out[(scen, rr)] = m.predict_proba(X[(rr, scen, k)][0][te])[:, 1]
        return rep, k, r, te, out
    res = Parallel(n_jobs=a.jobs)(delayed(job)(rep, k, r, i, tr, te) for rep in reps for k in range(4) for r, i, tr, te in folds)
    P = {}
    for rep, k, r, te, out in res:
        for (scen, rr), p_ in out.items():
            P.setdefault((scen, rr, k), np.zeros((5, len(pts))))[r, te] = p_
    P = {key: v.mean(0) for key, v in P.items()}
    pd.DataFrame([dict(patient_id=pts[i], scenario=s, representation=r_, landmark=T2[k], p=v[i], pcr=y[i]) for (s, r_, k), v in P.items()
                  for i in range(len(pts))]).to_csv(run / 'oof_predictions.csv', index=False)
    rng = np.random.default_rng(SEED)
    Bi = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(NB)) if y[i].min() != y[i].max()]
    bauc = lambda p_: np.array([roc_auc_score(y[i], p_[i]) for i in Bi])  # noqa: E731

    rows = []
    for (s, r_, k), p_ in P.items():
        if s != 'natural':
            continue
        b = bauc(p_)
        rows.append(dict(representation=r_, landmark=T2[k], n=len(y), n_new_mri=sum(obs[p][T2[k]] for p in pts), **metrics(y, p_),
                         auroc_lo=np.quantile(b, .025), auroc_hi=np.quantile(b, .975)))
    LM = pd.DataFrame(rows); LM.to_csv(run / 'landmark_metrics.csv', index=False)

    def pdiff(s1, r1, k1, s2, r2, k2, label, family):
        a_, b_ = P[(s1, r1, k1)], P[(s2, r2, k2)]
        d = bauc(a_) - bauc(b_)
        return dict(family=family, comparison=label, a=f'{r1}@{T2[k1]}', b=f'{r2}@{T2[k2]}', d_auroc=roc_auc_score(y, a_) - roc_auc_score(y, b_),
                    lo=np.quantile(d, .025), hi=np.quantile(d, .975), p_le0=float(np.mean(d <= 0)), primary=k1 == 3)
    cmp = [pdiff('natural', 'S', k, 'natural', 'B', 0, f'H1 S - B at {T2[k]}', 'H1 longitudinal information') for k in (1, 2, 3)]
    cmp += [pdiff('natural', 'H', k, 'natural', 'B', 0, f'H1 H - B at {T2[k]}', 'H1 longitudinal information') for k in (1, 2, 3)]
    for other in [r for r in ('H', 'M', 'Htab') if ('natural', r, 0) in P]:
        cmp += [pdiff('natural', 'S', k, 'natural', other, k, f'H2 S - {other} at {T2[k]}', 'H2 state maintenance') for k in range(4)]
    cmp += [pdiff('natural', 'SA', k, 'natural', 'S', k, f'arm context at {T2[k]}', 'secondary') for k in range(4)]
    CMP = pd.DataFrame(cmp); CMP.to_csv(run / 'paired_comparisons.csv', index=False)

    # ------------------------------------------------------------------ missing-visit robustness
    sc = [('miss_T1', 3, 'T1 missing @T3'), ('miss_T2', 3, 'T2 missing @T3'), ('miss_T1T2', 3, 'T1+T2 missing @T3'), ('miss_T2', 2, 'latest T2 missing @T2')]
    mr = []
    for scen, k, lab in sc:
        for r_ in ('H', 'S', 'M'):
            if (scen, r_, k) not in P:
                continue
            full, deg = P[('natural', r_, k)], P[(scen, r_, k)]
            d = bauc(full) - bauc(deg)
            mr.append(dict(scenario=lab, representation=r_, auroc_complete=roc_auc_score(y, full), auroc_degraded=roc_auc_score(y, deg),
                           drop=roc_auc_score(y, full) - roc_auc_score(y, deg), drop_lo=np.quantile(d, .025), drop_hi=np.quantile(d, .975)))
        for other in ('H', 'M'):
            if (scen, other, k) in P:
                d = bauc(P[(scen, 'S', k)]) - bauc(P[(scen, other, k)])
                mr.append(dict(scenario=lab, representation=f'S-{other}', auroc_degraded=roc_auc_score(y, P[(scen, 'S', k)]) - roc_auc_score(y, P[(scen, other, k)]),
                               drop_lo=np.quantile(d, .025), drop_hi=np.quantile(d, .975)))
    nat = np.array([not all(obs[p].values()) for p in pts])
    for r_ in ('H', 'S', 'M', 'B'):
        if ('natural', r_, 3) in P:
            mr.append(dict(scenario=f'natural-missingness subgroup @T3 (n={int(nat.sum())})', representation=r_,
                           auroc_degraded=roc_auc_score(y[nat], P[('natural', r_, 3)][nat])))
    pd.DataFrame(mr).to_csv(run / 'missing_visit_robustness.csv', index=False)

    # ------------------------------------------------------------------ revision mechanism (controlled)
    has1 = np.array([obs[p]['T1'] for p in pts])
    rv = []
    for r_ in ('H', 'S', 'ND', 'M'):
        for k in (1, 2, 3):
            key = ('revision', r_, k)
            if key not in P:
                continue
            base = P[('natural', 'S' if r_ == 'ND' else r_, k)]
            dev = np.abs(P[key] - base)[has1]
            tv = []
            for j, p in enumerate(pts):
                if not has1[j]:
                    continue
                Xr, cols = X[(r_, 'revision', k)]; Xc, _ = X[('S' if r_ == 'ND' else r_, 'natural', k)]
                c = cols.index('T1_ftv_pe')
                tv.append(abs(Xr[j, c] - Xc[j, c]) < 1e-9 if not np.isnan(Xc[j, c]) else np.isnan(Xr[j, c]))
            rv.append(dict(representation=r_, landmark=T2[k], n=int(has1.sum()), t1_value_correct=float(np.mean(tv)),
                           mean_abs_forecast_deviation=float(dev.mean()), recovered_lt_0p01=float((dev < .01).mean()),
                           auroc_revision=roc_auc_score(y[has1], P[key][has1]), auroc_clean=roc_auc_score(y[has1], base[has1])))
    pd.DataFrame(rv).to_csv(run / 'revision_mechanism.csv', index=False)

    # ------------------------------------------------------------------ external I-SPY1
    y1 = np.array([E.y(p) for p in E.pts])
    c2 = {p: {k_: clin[p][k_] for k_ in ('HR', 'HER2', 'age')} for p in pts}
    c1 = {p: E.clinical(p) for p in E.pts}
    st1 = {p: E.states(p) for p in E.pts}
    m1 = {} if a.no_memory else load_memory('ispy1', 'natural')
    ext, ext_pred = [], []
    rng1 = np.random.default_rng(SEED)
    B1 = [i for i in (rng1.integers(0, len(y1), len(y1)) for _ in range(NB)) if y1[i].min() != y1[i].max()]
    for k in range(4):
        pe = {}
        for r_ in ('B', 'H', 'S', 'M'):
            if r_ == 'M' and (not m1 or not mem['natural']):
                continue
            kk = 0 if r_ == 'B' else k
            def ft2(p):
                if r_ == 'M':
                    return F(table_from_memory(mem['natural'][p]['snapshots'][kk], kk, LD), kk, c2[p], names=LD)
                tab = I.table_from_state(st['natural'][p][kk], kk) if r_ == 'S' else I.table_from_source(p, 'natural', kk)
                return F({t: (None if v is None else dict(ld=v['ld'], day=v['day'])) for t, v in tab.items()}, kk, c2[p], names=LD)
            def ft1(p):
                if r_ == 'M':
                    return F(table_from_memory(m1[p]['snapshots'][kk], kk, LD), kk, c1[p], names=LD)
                tab = E.table_from_state(st1[p][kk], kk) if r_ == 'S' else E.table_from_source(p, kk)
                return F(tab, kk, c1[p], names=LD)
            Xtr = pd.DataFrame([ft2(p) for p in pts]); Xte = pd.DataFrame([ft1(p) for p in E.pts])[Xtr.columns]
            Xtr['day_latest'] = np.nan; Xte['day_latest'] = np.nan              # study day not available in I-SPY1
            m = fit(Xtr.to_numpy(float), y); p1 = m.predict_proba(Xte.to_numpy(float))[:, 1]; pe[r_] = p1
            b = np.array([roc_auc_score(y1[i], p1[i]) for i in B1])
            ext.append(dict(landmark=T2[k], representation=r_, n=len(y1), **metrics(y1, p1), auroc_lo=np.quantile(b, .025), auroc_hi=np.quantile(b, .975)))
        ext_pred += [dict(patient_id=E.pts[i], landmark=T2[k], representation=r_, p=pe[r_][i], pcr=y1[i]) for r_ in pe for i in range(len(y1))]
        if k:                                       # descriptive: external value of new visits
            d = np.array([roc_auc_score(y1[i], pe['S'][i]) - roc_auc_score(y1[i], pe['B'][i]) for i in B1])
            ext.append(dict(landmark=T2[k], representation='S-B (post hoc)', n=len(y1), auroc=roc_auc_score(y1, pe['S']) - roc_auc_score(y1, pe['B']),
                            auroc_lo=np.quantile(d, .025), auroc_hi=np.quantile(d, .975)))
        for other in ('H', 'M'):
            if other in pe:
                d = np.array([roc_auc_score(y1[i], pe['S'][i]) - roc_auc_score(y1[i], pe[other][i]) for i in B1])
                ext.append(dict(landmark=T2[k], representation=f'S-{other}', n=len(y1), auroc=roc_auc_score(y1, pe['S']) - roc_auc_score(y1, pe[other]),
                                auroc_lo=np.quantile(d, .025), auroc_hi=np.quantile(d, .975)))
    pd.DataFrame(ext).to_csv(run / 'external_validation.csv', index=False)
    pd.DataFrame(ext_pred).to_csv(run / 'external_predictions.csv', index=False)
    (run / 'run_info.json').write_text(json.dumps(dict(seeds=dict(cv='0-4', bootstrap=SEED), n_ispy2=len(pts), n_ispy1=len(E.pts),
                                                      memory_included=not a.no_memory), indent=1))
    pd.set_option('display.width', 250)
    for n_ in ('landmark_metrics', 'paired_comparisons', 'missing_visit_robustness', 'revision_mechanism', 'external_validation'):
        print(f'== {n_}'); print(pd.read_csv(run / f'{n_}.csv').round(4).to_string(index=False))
    print('done', run)


if __name__ == '__main__':
    main()
