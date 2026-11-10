#!/usr/bin/env python3
"""Experiment A main run (pre-specified analysis plan). Decision quantities first (no expert label), then calibration and scoring.
    python run.py [--lam 1.0]   -> results/decisions[_lamX].csv, results/revision[_lamX].csv, results/meta[_lamX].json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import solver as S

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
SEED = 20261026


def sigma_model(sc):
    """label-free scale of the inter-pipeline disagreement, 4 quantile bins of u_HD per compartment."""
    x = sc[sc.role.isin(['postop', 'followup']) & sc.hd_ce.notna() & sc.db_ce.notna()]
    model = {}
    for c in ('ce', 't2'):
        uh, ud = S.to_u(x[f'hd_{c}']), S.to_u(x[f'db_{c}'])
        edges = np.quantile(uh, [.25, .5, .75])
        b = np.searchsorted(edges, uh, side='right')
        med = [np.median(np.abs(ud - uh)[b == k]) if (b == k).any() else np.nan for k in range(4)]
        med = pd.Series(med).bfill().ffill().to_numpy()                   # an empty bin (no scan can fall in it) takes its neighbour
        model[c] = (edges, np.maximum(med, .05))
    return model


def sig(model, c, u):
    edges, s = model[c]
    return s[np.searchsorted(edges, np.atleast_1d(u), side='right')]


def decisions_of(sc, model, lam, pipe='hd', revise=None):
    """all evaluable decisions; revise = (patient, tp) whose HD measurement is replaced by DeepBraTumIA."""
    out = []
    for p, g in sc.groupby('patient', sort=True):
        g = g.sort_values('week')
        val = {}
        for r in g.itertuples():
            src = 'db' if (pipe == 'db' or (revise is not None and (p, r.tp) == revise)) else 'hd'
            val[r.tp] = {c: getattr(r, f'{src}_{c}') for c in ('ce', 't2')}
        pri_rows = g[g.prior_ok]
        for r in g[g.evaluable].itertuples():
            pr = [q for q in pri_rows.itertuples() if q.week < r.week and all(np.isfinite(val[q.tp][c]) for c in ('ce', 't2'))]
            if not pr or not all(np.isfinite(val[r.tp][c]) for c in ('ce', 't2')):
                continue
            m_t = {c: float(S.to_u(val[r.tp][c])) for c in ('ce', 't2')}
            m_p = {c: S.to_u([val[q.tp][c] for q in pr]) for c in ('ce', 't2')}
            d = S.Decision(key=(p, r.tp), m_t=m_t, s_t={c: float(sig(model, c, m_t[c])[0]) for c in m_t}, m_p=m_p,
                           s_p={c: sig(model, c, m_p[c]) for c in m_p}, lt3m=bool(r.lt3m), lam=lam)
            out.append((d, [q.tp for q in pr]))
    return out


def one(d):
    j = S.solve(d)
    f = S.solve(S.Decision(**{**d.__dict__, 'fixed_priors': True, 'cost': {}, 'delta': 0.0}))
    fc, fk = S.flip_cost(j)
    gc, gk = S.flip_cost(f)
    return dict(patient=d.key[0], tp=d.key[1], lt3m=int(d.lt3m), point=j.point(), delta=j.delta, flip=fc, flip_to=fk,
                flip_fixed=gc, flip_fixed_to=gk, **{f'cost_{k}': v for k, v in j.cost.items()},
                **{f'fcost_{k}': v for k, v in f.cost.items()})


def quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float)); k = math.ceil((len(s) + 1) * (1 - alpha))
    return float('inf') if k > len(s) else float(s[k - 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lam', type=float, default=1.0); a = ap.parse_args()
    tag = '' if a.lam == 1.0 else f'_lam{a.lam:g}'
    sc = pd.read_csv(RES / 'scans.csv')
    model = sigma_model(sc)
    D = decisions_of(sc, model, a.lam)
    rows = Parallel(n_jobs=32)(delayed(one)(d) for d, _ in D)
    df = pd.DataFrame(rows)
    # C2 uncertainty and C4 pipeline agreement (point rules)
    scx = sc.set_index(['patient', 'tp'])
    df['disagreement'] = [max(abs(float(S.to_u(scx.at[(p, t), f'hd_{c}']) - S.to_u(scx.at[(p, t), f'db_{c}']))) for c in ('ce', 't2'))
                          for p, t in zip(df.patient, df.tp)]
    Ddb = {d.key: d.point() for d, _ in decisions_of(sc, model, a.lam, pipe='db')}
    df['db_point'] = [Ddb.get((p, t), '') for p, t in zip(df.patient, df.tp)]
    # folds (patient level, fixed seed)
    pts = sorted(df.patient.unique()); perm = np.random.default_rng(SEED).permutation(pts)
    fold = {p: i % 5 for i, p in enumerate(perm)}
    df['fold'] = df.patient.map(fold)

    # ---------------------------------------------------------------- calibration (expert labels of other-fold patients only)
    lab = sc.set_index(['patient', 'tp']).rating
    y = np.array([lab[(p, t)] for p, t in zip(df.patient, df.tp)])
    s_joint = pd.Series([df.iloc[i][f'cost_{y[i]}'] for i in range(len(df))]).groupby(df.patient.values).max()
    s_fixed = pd.Series([df.iloc[i][f'fcost_{y[i]}'] for i in range(len(df))]).groupby(df.patient.values).max()
    for al in ALPHAS:
        qj = {k: quantile([s_joint[p] for p in pts if fold[p] != k], al) for k in range(5)}
        qf = {k: quantile([s_fixed[p] for p in pts if fold[p] != k], al) for k in range(5)}
        df[f'q_{al}'] = df.fold.map(qj); df[f'qf_{al}'] = df.fold.map(qf)
        df[f'cert_{al}'] = (df.flip > df[f'q_{al}'] + df.delta).astype(int)
        df[f'certf_{al}'] = (df.flip_fixed > df[f'qf_{al}']).astype(int)
        df[f'cover_{al}'] = (df.patient.map(s_joint) <= df[f'q_{al}']).astype(int)
    df['expert'] = y
    df['wrong'] = (df.point != df.expert).astype(int)
    df['s_patient'] = df.patient.map(s_joint)
    df.to_csv(RES / f'decisions{tag}.csv', index=False)

    # ---------------------------------------------------------------- A4 revision: each prior scan replaced by DeepBraTumIA
    rev = []
    base = df.set_index(['patient', 'tp'])
    for p, g in sc[sc.patient.isin(pts)].groupby('patient'):
        if base.loc[p].shape[0] < 2:
            continue
        cand = g[g.prior_ok & g.db_ce.notna() & g.db_t2.notna()]
        for s_ in cand.itertuples():
            Dr = [d for d, pri in decisions_of(sc[sc.patient == p], model, a.lam, revise=(p, s_.tp)) if s_.tp in pri or d.key[1] == s_.tp]
            if not Dr:
                continue
            for d in Dr:
                r = one(d); b = base.loc[(p, d.key[1])]
                q = b['q_0.1']; fresh = int(r['flip'] > q + r['delta'])
                rev.append(dict(patient=p, revised=s_.tp, decision=d.key[1], old_point=b.point, new_point=r['point'], old_cert=int(b['cert_0.1']),
                                new_cert=fresh, stale=int(b['cert_0.1'] == 1 and (fresh == 0 or r['point'] != b.point)),
                                expert=b.expert, n_patient_decisions=int(base.loc[p].shape[0])))
    R = pd.DataFrame(rev); R.to_csv(RES / f'revision{tag}.csv', index=False)
    meta = dict(lam=a.lam, n_decisions=len(df), n_patients=len(pts), sigma={c: model[c][1].tolist() for c in model},
                sigma_edges={c: model[c][0].tolist() for c in model}, folds=fold, seed=SEED, step=S.STEP)
    (RES / f'meta{tag}.json').write_text(json.dumps(meta, indent=1, default=str))
    print(f'decisions {len(df)} patients {len(pts)} revision rows {len(R)}')


if __name__ == '__main__':
    main()
