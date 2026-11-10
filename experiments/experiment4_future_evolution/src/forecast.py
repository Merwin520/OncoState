#!/usr/bin/env python3
"""4A future FTV forecasting + 4C treatment-conditioned FTV (experiment_plan.md). Writes results/4A_*.csv, results/4C_ftv_*.csv.
  python forecast.py [--jobs 48] [--perm 100]"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from check import check
from common import E29, ISPY2, NB, RES, REPO, SEED, T2, F, table_from_memory

warnings.filterwarnings('ignore', category=ConvergenceWarning)
FOLDS = REPO / 'outputs/longitudinal_forecasting/analysis/folds.csv'
NEXT = [('T0', 'T1'), ('T1', 'T2'), ('T2', 'T3')]
LONG = [('T0', 'T2'), ('T0', 'T3'), ('T1', 'T3')]
IDX = {t: i for i, t in enumerate(T2)}


def model(kind):
    if kind == 'ridge':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    g = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    return GridSearchCV(g, {'max_depth': [2, 3], 'learning_rate': [.03, .1]}, cv=3, scoring='neg_mean_absolute_error')


def best(m, kind):
    return m if kind == 'ridge' else HistGradientBoostingRegressor(max_iter=300, random_state=0, **m.best_params_)


# ----------------------------------------------------------------------------------------------------------------- features
def feats(tab, a, clin, dt, rep, arm=None, arms=None):
    """Inputs available at origin visit a (tab holds visits <= a only)."""
    last = [tab[t]['ftv_pe'] for t in T2[:IDX[a] + 1] if tab.get(t) and tab[t].get('ftv_pe') is not None]
    base = {'dt': dt}
    if rep == 'last':
        return {**base, 'last_log1p': np.log1p(last[-1]) if last else np.nan}
    if rep == 'clinical':
        return {**base, **{k: (np.nan if v is None else float(v)) for k, v in clin.items()}}
    if rep == 'armonly':
        return {**base, 'last_log1p': np.log1p(last[-1]) if last else np.nan, **{f'arm_{x}': float(arm == x) for x in arms}}
    names = ('ftv_pe',) if rep == 'ftvhist' else ('ftv_pe', 'ftv_ser', 'ld', 'sph', 'bpe')
    out = {**base, **F(tab, IDX[a], clin, names=names)}
    out['day_latest'] = np.nan if 'day_latest' not in out else out['day_latest']
    if rep == 'state_arm':
        out.update({f'arm_{x}': float(arm == x) for x in arms})
    return out


def load():
    I = ISPY2(); folds = pd.read_csv(FOLDS)
    pts = sorted(folds.patient_id.astype(str).unique())
    S, H, C, A, day, y = {}, {}, {}, {}, {}, {}
    for p in pts:
        st = I.states(p, 'natural')
        S[p] = [I.table_from_state(st[k], k) for k in range(4)]
        H[p] = [I.table_from_source(p, 'natural', k) for k in range(4)]
        c = I.clinical(p); A[p] = c.pop('arm'); C[p] = c
        full = S[p][3]
        day[p] = {t: (np.nan if (full[t] or {}).get('day') is None else float(full[t]['day'])) for t in T2}
        y[p] = {t: (np.log1p(full[t]['ftv_pe']) if full[t] and full[t].get('ftv_pe') is not None else np.nan) for t in T2}
    return I, folds, pts, S, H, C, A, day, y


def task_rows(pts, S, C, A, day, y, a, b, rep, arms, tables=None, dt_override=None):
    rows, ids = [], []
    for p in pts:
        if np.isnan(y[p][a]) or np.isnan(y[p][b]):
            continue
        tab = (tables or S)[p][IDX[a]]
        dt = dt_override(p) if dt_override else day[p][b] - day[p][a]
        rows.append(feats(tab, a, C[p], dt, rep, A[p], arms)); ids.append(p)
    X = pd.DataFrame(rows, index=ids)
    return X, np.array([y[p][b] for p in ids])


# ----------------------------------------------------------------------------------------------------------------- fitting
def fit_predict(Xtr, ytr, Xte, kind, conformal=False):
    if kind == 'mean':
        return np.full(len(Xte), ytr.mean()), np.nan
    m = model(kind).fit(Xtr.to_numpy(float), ytr)
    p = m.predict(Xte.to_numpy(float))
    q = np.nan
    if conformal:
        oof = cross_val_predict(best(m, kind) if kind == 'gbm' else model('ridge'), Xtr.to_numpy(float), ytr, cv=KFold(5, shuffle=True, random_state=0))
        q = float(np.quantile(np.abs(oof - ytr), .9))
    return p, q


CACHE = RES / 'cache'


def cached(key, fn, *args):
    """Per-task disk cache so that an interrupted run resumes where it stopped."""
    import pickle
    f = CACHE / (key.replace('>', '').replace('|', '_').replace('/', '_') + '.pkl')
    if f.exists():
        return pickle.loads(f.read_bytes())
    out = fn(*args); CACHE.mkdir(parents=True, exist_ok=True); f.write_bytes(pickle.dumps(out)); return out


def run_task(name, X, yv, kind, splits, conformal):
    """Out-of-fold predictions for one task over the 25 folds; returns (pred per repeat [5, n], conformal half-width per repeat)."""
    ids = list(X.index); pos = {p: i for i, p in enumerate(ids)}
    P = np.full((5, len(ids)), np.nan); Q = np.full((5, len(ids)), np.nan)
    for r, te_pts in splits:
        te = [pos[p] for p in te_pts if p in pos]; tr = [i for i in range(len(ids)) if i not in set(te)]
        if not te:
            continue
        p, q = fit_predict(X.iloc[tr], yv[tr], X.iloc[te], kind, conformal)
        P[r, te] = p; Q[r, te] = q
    return name, ids, P, Q


def metrics(y, p, q=None):
    yc, pc = np.expm1(y), np.expm1(p)
    d = dict(n=len(y), MAE_log=np.mean(np.abs(p - y)), MAE_cc=np.mean(np.abs(pc - yc)), RMSE_cc=np.sqrt(np.mean((pc - yc) ** 2)),
             MedAE_cc=np.median(np.abs(pc - yc)), spearman=spearmanr(y, p).statistic)
    if q is not None and np.all(np.isfinite(q)):
        lo, hi = p - q, p + q
        d.update(coverage90=np.mean((y >= lo) & (y <= hi)), width90_cc_median=np.median(np.expm1(hi) - np.maximum(np.expm1(lo), 0)))
    return d


def boot_idx(n, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(NB)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=48); ap.add_argument('--perm', type=int, default=100); a = ap.parse_args()
    check()
    I, folds, pts, S, H, C, A, day, y = load()
    arms = sorted(set(A.values()))
    splits = [(r, list(g.patient_id.astype(str))) for (r, f), g in folds.groupby(['repeat', 'fold'])]
    # ---------------------------------------------------------------- task list
    tasks = []
    for (o, t) in NEXT:
        for rep in ('last', 'ftvhist', 'state', 'state_arm', 'armonly', 'clinical'):
            for kind in ('ridge', 'gbm'):
                tasks.append((f'{o}->{t}|{rep}|{kind}', o, t, rep, kind))
        tasks.append((f'{o}->{t}|state|mean', o, t, 'state', 'mean'))
    for (o, t) in LONG:
        for rep in ('ftvhist', 'state'):
            for kind in ('ridge', 'gbm'):
                tasks.append((f'{o}->{t}|{rep}|{kind}', o, t, rep, kind))
    data = {n: task_rows(pts, S, C, A, day, y, o, t, rep, arms) for n, o, t, rep, kind in tasks}
    res = Parallel(n_jobs=a.jobs)(delayed(cached)(f'task_{n}', run_task, n, *data[n], kind, splits, kind in ('ridge', 'gbm') and rep in ('state', 'ftvhist'))
                                  for n, o, t, rep, kind in tasks)
    oof = []
    for n, ids, P, Q in res:
        for i, p in enumerate(ids):
            oof.append(dict(task=n, patient_id=p, y=data[n][1][i], pred=np.nanmean(P[:, i]), q90=np.nanmean(Q[:, i])))
    # LOCF (carry the last observed FTV forward)
    for (o, t) in NEXT + LONG:
        X, yv = task_rows(pts, S, C, A, day, y, o, t, 'last', arms)
        oof += [dict(task=f'{o}->{t}|last|locf', patient_id=p, y=yv[i], pred=X.at[p, 'last_log1p'], q90=np.nan) for i, p in enumerate(X.index)]
    O = pd.DataFrame(oof); O.to_csv(RES / '4A_oof.csv', index=False)

    # ---------------------------------------------------------------- autoregressive rollout (ftvhist GBM next-visit models, refitted per fold)
    med = {t: float(np.nanmedian([day[p][t] for p in pts])) for t in T2}
    roll = Parallel(n_jobs=a.jobs)(delayed(cached)(f'roll_{i}', rollout_fold, r, te, pts, S, C, A, day, y, arms, med) for i, (r, te) in enumerate(splits))
    R = pd.DataFrame([x for rr in roll for x in rr]).groupby(['task', 'patient_id']).mean(numeric_only=True).reset_index()
    O = pd.concat([O, R], ignore_index=True); O.to_csv(RES / '4A_oof.csv', index=False)

    # ---------------------------------------------------------------- metrics, paired differences, subgroups
    met, pair = [], []
    t0 = {p: y[p]['T0'] for p in pts}; terc = np.nanquantile(list(t0.values()), [1 / 3, 2 / 3])
    for task, g in O.groupby('task'):
        q = g.q90.to_numpy() if g.q90.notna().all() else None
        m = metrics(g.y.to_numpy(), g.pred.to_numpy(), q); bi = boot_idx(len(g))
        e = np.abs(g.pred.to_numpy() - g.y.to_numpy()); ec = np.abs(np.expm1(g.pred.to_numpy()) - np.expm1(g.y.to_numpy()))
        m.update(MAE_log_lo=np.percentile([e[i].mean() for i in bi], 2.5), MAE_log_hi=np.percentile([e[i].mean() for i in bi], 97.5),
                 MAE_cc_lo=np.percentile([ec[i].mean() for i in bi], 2.5), MAE_cc_hi=np.percentile([ec[i].mean() for i in bi], 97.5))
        tr, rep, kind = task.split('|'); met.append(dict(task=task, transition=tr, representation=rep, model=kind, subgroup='all', **m))
        tt = np.digitize([t0[p] for p in g.patient_id], terc)
        for k, lab in enumerate(['T0 FTV low', 'T0 FTV mid', 'T0 FTV high']):
            gg = g[tt == k]
            if len(gg) > 10:
                met.append(dict(task=task, transition=tr, representation=rep, model=kind, subgroup=lab, **metrics(gg.y.to_numpy(), gg.pred.to_numpy())))
    M = pd.DataFrame(met); M.to_csv(RES / '4A_metrics.csv', index=False)

    def pdiff(ta, tb, label, family):
        ga, gb = O[O.task == ta].set_index('patient_id'), O[O.task == tb].set_index('patient_id')
        com = ga.index.intersection(gb.index); ga, gb = ga.loc[com], gb.loc[com]
        ea, eb = np.abs(ga.pred - ga.y).to_numpy(), np.abs(gb.pred - gb.y).to_numpy()
        ca, cb = np.abs(np.expm1(ga.pred) - np.expm1(ga.y)).to_numpy(), np.abs(np.expm1(gb.pred) - np.expm1(gb.y)).to_numpy()
        bi = boot_idx(len(com)); d = [(ea[i] - eb[i]).mean() for i in bi]; dc = [(ca[i] - cb[i]).mean() for i in bi]
        return dict(family=family, comparison=label, a=ta, b=tb, n=len(com), dMAE_log=(ea - eb).mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                    p_ge0=float(np.mean(np.array(d) >= 0)), dMAE_cc=(ca - cb).mean(), lo_cc=np.percentile(dc, 2.5), hi_cc=np.percentile(dc, 97.5))
    for o, t in NEXT:
        tr = f'{o}->{t}'
        pair += [pdiff(f'{tr}|state|gbm', f'{tr}|last|locf', f'{tr} GBM[state] - LOCF', '4A primary'),
                 pdiff(f'{tr}|state|gbm', f'{tr}|state|mean', f'{tr} GBM[state] - mean', '4A baseline'),
                 pdiff(f'{tr}|state|gbm', f'{tr}|last|gbm', f'{tr} GBM: state - last (representation)', '4A representation'),
                 pdiff(f'{tr}|state|ridge', f'{tr}|last|ridge', f'{tr} Ridge: state - last (representation)', '4A representation'),
                 pdiff(f'{tr}|state|gbm', f'{tr}|state|ridge', f'{tr} state: GBM - Ridge (model)', '4A model'),
                 pdiff(f'{tr}|state_arm|ridge', f'{tr}|state|ridge', f'{tr} Ridge: state+arm - state', '4C arm'),
                 pdiff(f'{tr}|state_arm|gbm', f'{tr}|state|gbm', f'{tr} GBM: state+arm - state', '4C arm'),
                 pdiff(f'{tr}|armonly|ridge', f'{tr}|state|ridge', f'{tr} Ridge: arm+last - state', '4C arm'),
                 pdiff(f'{tr}|clinical|gbm', f'{tr}|state|gbm', f'{tr} GBM: clinical only - state', '4C clinical')]
    for o, t in LONG:
        tr = f'{o}->{t}'
        pair += [pdiff(f'{tr}|ftvhist|gbm', f'{tr}|last|locf', f'{tr} direct GBM[ftvhist] - LOCF', '4A horizon'),
                 pdiff(f'{tr}|rollout|gbm', f'{tr}|ftvhist|gbm', f'{tr} rollout - direct', '4A horizon')]
    # sequential incorporation: same target T3, origins T2 < T1 < T0 (fixed cohort = patients present in all three tasks)
    for later, earlier in (('T2->T3', 'T1->T3'), ('T1->T3', 'T0->T3'), ('T2->T3', 'T0->T3')):
        pair.append(pdiff(f'{later}|ftvhist|gbm', f'{earlier}|ftvhist|gbm', f'target T3: origin {later[:2]} - origin {earlier[:2]}', '4A sequential'))
    for later, earlier in (('T1->T2', 'T0->T2'),):
        pair.append(pdiff(f'{later}|ftvhist|gbm', f'{earlier}|ftvhist|gbm', f'target T2: origin {later[:2]} - origin {earlier[:2]}', '4A sequential'))
    PR = pd.DataFrame(pair); PR.to_csv(RES / '4A_paired.csv', index=False)

    # ---------------------------------------------------------------- 4C arm permutation (Ridge, within-subtype, refit)
    sub = {p: f"{C[p]['HR']}{C[p]['HER2']}" for p in pts}
    perm = []
    obs = {r.comparison: r.dMAE_log for r in PR.itertuples() if 'Ridge: state+arm - state' in r.comparison}
    base_tasks = {f'{o}->{t}': O[O.task == f'{o}->{t}|state|ridge'].set_index('patient_id') for o, t in NEXT}
    out = Parallel(n_jobs=a.jobs)(delayed(cached)(f'perm_{k}_{o}{t}', perm_once, k, o, t, pts, S, C, A, day, y, arms, splits, sub, base_tasks[f'{o}->{t}'])
                                  for k in range(a.perm) for o, t in NEXT)
    for (k, tr, d) in out:
        perm.append(dict(perm=k, transition=tr, dMAE_log=d))
    PM = pd.DataFrame(perm); PM.to_csv(RES / '4C_ftv_arm_permutation.csv', index=False)
    pv = {tr: float((1 + (g.dMAE_log <= obs[f'{tr} Ridge: state+arm - state']).sum()) / (1 + len(g))) for tr, g in PM.groupby('transition')}
    # ---------------------------------------------------------------- state-maintenance input equality (S vs H vs M)
    eq = state_equality(pts, S, H, C, A, day, y, arms)
    (RES / '4A_run_info.json').write_text(json.dumps(dict(arm_perm_p_one_sided=pv, state_input_equality=eq, protocol_median_days=med,
                                                          n_tasks=len(tasks), folds=str(FOLDS)), indent=1, default=float))
    pd.set_option('display.width', 250)
    print(M[M.subgroup == 'all'][['task', 'n', 'MAE_log', 'MAE_cc', 'RMSE_cc', 'spearman', 'coverage90']].round(3).to_string(index=False))
    print(PR[['comparison', 'n', 'dMAE_log', 'lo', 'hi', 'dMAE_cc']].round(4).to_string(index=False))
    print('arm permutation p', pv); print('state equality', eq)


def rollout_fold(r, te_pts, pts, S, C, A, day, y, arms, med):
    """Fit ftvhist-GBM next-visit models on the training patients of this fold and roll test patients forward without real observations."""
    te = set(te_pts); tr_pts = [p for p in pts if p not in te]
    mods = {}
    for o, t in NEXT:
        X, yv = task_rows(tr_pts, S, C, A, day, y, o, t, 'ftvhist', arms)
        mods[(o, t)] = (model('gbm').fit(X.to_numpy(float), yv), list(X.columns))
    rows = []
    for o, t in LONG:
        for p in te_pts:
            if np.isnan(y[p][o]) or np.isnan(y[p][t]):
                continue
            tab = {k: (dict(v) if v else None) for k, v in S[p][IDX[o]].items()}
            cur, cur_day = o, day[p][o]
            while cur != t:
                nxt = T2[IDX[cur] + 1]
                tday = day[p][t] if nxt == t else med[nxt]
                m, cols = mods[(cur, nxt)]
                x = pd.DataFrame([feats(tab, cur, C[p], tday - cur_day, 'ftvhist', A[p], arms)])[cols]
                pr = float(m.predict(x.to_numpy(float))[0])
                tab[nxt] = dict(day=tday, ftv_pe=float(np.expm1(pr)), ftv_ser=None, ld=None, sph=None, bpe=None)
                cur, cur_day = nxt, tday
            rows.append(dict(task=f'{o}->{t}|rollout|gbm', patient_id=p, repeat=r, y=y[p][t], pred=pr))
    return rows


def perm_once(k, o, t, pts, S, C, A, day, y, arms, splits, sub, base):
    rng = np.random.default_rng(1000 + k)
    Ap = dict(A)
    for s in set(sub.values()):
        g = [p for p in pts if sub[p] == s]; sh = list(rng.permutation([A[p] for p in g]))
        Ap.update(dict(zip(g, sh)))
    X, yv = task_rows(pts, S, C, Ap, day, y, o, t, 'state_arm', arms)
    _, ids, P, _ = run_task('perm', X, yv, 'ridge', splits, False)
    pred = np.nanmean(P, 0)
    b = base.loc[ids]
    return k, f'{o}->{t}', float(np.mean(np.abs(pred - yv)) - np.mean(np.abs(b.pred.to_numpy() - b.y.to_numpy())))


def state_equality(pts, S, H, C, A, day, y, arms):
    import json as _j
    mem = {}
    for l in (REPO / 'outputs/longitudinal_forecasting/raw/memory_ispy2_natural.jsonl').read_text().splitlines():
        j = _j.loads(l); mem[j['patient_id']] = j
    Mt = {p: [table_from_memory(mem[p]['snapshots'][k], k, ('ftv_pe', 'ftv_ser', 'ld', 'sph', 'bpe')) for k in range(4)] for p in pts if p in mem}
    out = {}
    for o, t in NEXT:
        Xs, _ = task_rows(pts, S, C, A, day, y, o, t, 'state', arms)
        for name, tabs in (('H', H), ('M', Mt)):
            Xo, _ = task_rows([p for p in pts if p in tabs], tabs, C, A, day, y, o, t, 'state', arms)
            com = Xs.index.intersection(Xo.index); a_, b_ = Xs.loc[com].to_numpy(float), Xo.loc[com, Xs.columns].to_numpy(float)
            same = np.all(np.isclose(a_, b_, equal_nan=True, atol=1e-9), axis=1)
            out[f'{o}->{t} S vs {name}'] = dict(n=int(len(com)), identical_rows=int(same.sum()))
    return out


if __name__ == '__main__':
    main()
