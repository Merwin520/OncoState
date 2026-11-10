#!/usr/bin/env python3
"""Exp 2.7b: treatment-conditioned transitions of the Exp 2.8 persistent twins (I-SPY2), as pre-specified in the analysis plan.

Reads every twin version from experiments/exp28_repeated_synchronization/patient_states, builds transitions t -> t+1 with S_t taken
from version v_t only, and compares S_t + dt with S_t + dt + randomized arm on identical patient-grouped folds.
Writes outputs/exp27/ispy2/run/ (transitions.csv, oof.csv, results.json, experiment_table.csv).
  python run_ispy2.py [--perm 200] [--boot 5000] [--jobs 100] [--quick]
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / 'outputs/exp27/ispy2'
STATES = REPO / 'experiments/exp28_repeated_synchronization/patient_states'
CLIN = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'exp28_work/metadata/source/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx'))
T = ['T0', 'T1', 'T2', 'T3']
EPS = 0.1
STATE = ['log_ftv', 'log_ld', 'sphericity', 'bpe', 'log_ftv0', 'prev_logratio', 'is_t0', 'tr1', 'tr2', 'HR', 'HER2', 'MP', 'age', 'postmeno']
CONTROL = {'Paclitaxel', 'Paclitaxel + Trastuzumab'}
SUBTYPES = ['HR-/HER2- (TN)', 'HR+/HER2-', 'HR+/HER2+', 'HR-/HER2+']


# ------------------------------------------------------------------------------------------------- data from the twins
def _val(h, t, k):
    f = h.get(t, {}).get('fields', {}).get(k, {})
    return f.get('value') if f.get('available') else None


def load():
    rows = []
    for pdir in sorted(p for p in STATES.iterdir() if p.is_dir()):
        vers = {json.loads(f.read_text())['current_timepoint']: json.loads(f.read_text()) for f in pdir.glob('v*_T*.json')}
        for k in range(3):
            a, b = T[k], T[k + 1]
            if a not in vers or b not in vers:
                continue
            va, vb = vers[a], vers[b]                                   # S_t from v_t only; target from v_{t+1}
            h = va['history']
            fa, fb = _val(h, a, 'ftv_pe_cc'), _val(vb['history'], b, 'ftv_pe_cc')
            if h.get(a, {}).get('status') != 'observed' or vb['history'].get(b, {}).get('status') != 'observed' or fa is None or fb is None:
                continue
            f0 = _val(h, 'T0', 'ftv_pe_cc')
            prev = None
            if k > 0:
                fp = _val(h, T[k - 1], 'ftv_pe_cc')
                prev = np.log((fa + EPS) / (fp + EPS)) if fp is not None else None
            c, tc = va['clinical_context'], va['treatment_context']
            ld = _val(h, a, 'longest_diameter_cm')
            rows.append(dict(patient_id=str(va['patient_id']), transition=f'{a}->{b}', k=k,
                             log_ftv=np.log(fa + EPS), log_ld=np.log(ld + .1) if ld is not None else np.nan,
                             sphericity=_val(h, a, 'sphericity'), bpe=_val(h, a, 'bpe_5slice_mean'),
                             log_ftv0=np.log(f0 + EPS) if f0 is not None else np.nan,
                             prev_logratio=prev if prev is not None else 0.0, is_t0=int(k == 0), tr1=int(k == 1), tr2=int(k == 2),
                             HR=c.get('HR'), HER2=c.get('HER2'), MP=c.get('MP'), age=c.get('age_at_screening'),
                             postmeno=int(str(c.get('menopausal_status', '')).startswith('Post')), subtype=c.get('clinical_subtype'),
                             arm=tc['arm'], exp=int(tc['arm'] not in CONTROL),
                             dt=vb['history'][b]['study_day_offset'] - h[a]['study_day_offset'],
                             y=np.log((fb + EPS) / (fa + EPS))))
    df = pd.DataFrame(rows).astype({c: float for c in STATE + ['dt', 'y']})
    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------- design (fitted in-fold)
def rcs_basis(x, k):
    x = np.asarray(x, float); d = k[-1] - k[-2]; t = lambda z: np.maximum(z, 0) ** 3  # noqa: E731
    return np.column_stack([x] + [(t(x - k[j]) - t(x - k[-2]) * (k[-1] - k[j]) / d + t(x - k[-1]) * (k[-2] - k[j]) / d) / (k[-1] - k[0]) ** 2
                                  for j in range(len(k) - 2)])


class Design(BaseEstimator, TransformerMixin):
    def __init__(self, arms=(), arm_onehot=False, arm_x_subtype=False, exp=False, exp_x_subtype=False):
        self.arms, self.arm_onehot, self.arm_x_subtype, self.exp, self.exp_x_subtype = arms, arm_onehot, arm_x_subtype, exp, exp_x_subtype

    def fit(self, X, y=None):
        self.med_ = X[STATE].median(); self.miss_ = [c for c in STATE if X[c].isna().any()]
        self.knots_ = np.quantile(X.dt, [.05, .35, .65, .95])
        return self

    def transform(self, X):
        parts = [X[STATE].fillna(self.med_).to_numpy(float), X[self.miss_].isna().to_numpy(float), rcs_basis(X.dt, self.knots_)]
        oh = np.column_stack([(X.arm == a).to_numpy(float) for a in self.arms])
        sb = np.column_stack([(X.subtype == s).to_numpy(float) for s in SUBTYPES])
        if self.arm_onehot:
            parts.append(oh)
        if self.arm_x_subtype:
            parts.append(np.column_stack([oh[:, i] * sb[:, j] for i in range(oh.shape[1]) for j in range(4)]))
        if self.exp:
            parts.append(X[['exp']].to_numpy(float))
        if self.exp_x_subtype:
            parts.append(X[['exp']].to_numpy(float) * sb)
        return np.column_stack(parts)


def model(kind, design, hgb=False):
    if hgb:
        return make_pipeline(Design(**design), HistGradientBoostingRegressor(loss='absolute_error', random_state=0)), {
            'histgradientboostingregressor__max_depth': [2, 3], 'histgradientboostingregressor__max_iter': [100, 300],
            'histgradientboostingregressor__learning_rate': [.05], 'histgradientboostingregressor__min_samples_leaf': [20]}
    return make_pipeline(Design(**design), StandardScaler(), ElasticNet(max_iter=50000)), {
        'elasticnet__alpha': [.001, .003, .01, .03, .1, .3], 'elasticnet__l1_ratio': [.05, .5, .95]}


# ------------------------------------------------------------------------------------------------- CV
def folds(df):
    out = []
    for r in range(5):
        sgk = StratifiedGroupKFold(5, shuffle=True, random_state=r)
        for f, (tr, te) in enumerate(sgk.split(df, df.arm, df.patient_id)):
            out.append(dict(repeat=r, fold=f, train=tr, test=te))
    return out


def fit_one(df, fd, design, hgb=False, arm_map=None, fake=None, dt_seed=None, y=None):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    X = df.copy()
    if y is not None:
        X['y'] = y
    tr, te = X.iloc[fd['train']].copy(), X.iloc[fd['test']].copy()
    if arm_map is not None:                                           # permuted arms for training patients only
        tr['arm'] = tr.patient_id.map(arm_map); tr['exp'] = (~tr.arm.isin(CONTROL)).astype(int)
    if fake is not None:                                              # fake arms for everyone
        for d_ in (tr, te):
            d_['arm'] = d_.patient_id.map(fake); d_['exp'] = (~d_.arm.isin(CONTROL)).astype(int)
    if dt_seed is not None:
        tr['dt'] = np.random.default_rng(dt_seed).permutation(tr.dt.to_numpy())
    p, grid = model(None, design, hgb)
    inner = list(GroupKFold(3).split(tr, tr.y, tr.patient_id))
    gs = GridSearchCV(p, grid, cv=inner, scoring='neg_mean_absolute_error', n_jobs=1).fit(tr, tr.y.to_numpy())
    return pd.DataFrame(dict(row=fd['test'], repeat=fd['repeat'], y_true=te.y.to_numpy(), y_pred=gs.predict(te)))


def avg(oof, n):
    g = oof.groupby('row').y_pred.mean()
    return g.reindex(range(n)).to_numpy()


def paired(df, oof_b, oof_a, n_boot, mask=None, seed=0):
    n = len(df); pb, pa = avg(oof_b, n), avg(oof_a, n)
    m = np.ones(n, bool) if mask is None else mask
    eb, ea = np.abs(df.y.to_numpy() - pb), np.abs(df.y.to_numpy() - pa)
    d = (eb - ea)[m]; pid = df.patient_id.to_numpy()[m]
    up = np.unique(pid); idx = {p: np.where(pid == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.choice(up, len(up)); bs.append(np.concatenate([d[idx[p]] for p in s]).mean())
    bs = np.array(bs)
    reps = []
    for r in range(5):
        b_, a_ = oof_b[oof_b.repeat == r].set_index('row'), oof_a[oof_a.repeat == r].set_index('row')
        rr = [i for i in b_.index if m[i]]
        reps.append(float((b_.loc[rr].y_true - b_.loc[rr].y_pred).abs().mean() - (a_.loc[rr].y_true - a_.loc[rr].y_pred).abs().mean()))
    return dict(mae_b=float(eb[m].mean()), mae_a=float(ea[m].mean()), dmae=float(d.mean()), rel_pct=float(100 * d.mean() / ea[m].mean()),
                ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                p_boot=float(min(1, 2 * min((bs >= 0).mean(), (bs <= 0).mean()))), repeats_negative=int(sum(x < 0 for x in reps)),
                per_repeat=reps, n_transitions=int(m.sum()), n_patients=int(len(up)))


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (len(ps) - i) * ps[k])); adj[k] = run
    return adj.tolist()


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perm', type=int, default=200); ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--jobs', type=int, default=100); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--run-name', default='run', help='output folder under outputs/exp27/ispy2')
    a = ap.parse_args()
    if a.quick:
        a.perm, a.boot = 3, 300
    df = load(); n = len(df)
    arms = tuple(sorted(df.arm.unique()))
    FD = folds(df)
    run = OUT / a.run_name; run.mkdir(parents=True, exist_ok=True)
    df.to_csv(run / 'transitions.csv', index=False)
    P = Parallel(n_jobs=a.jobs)
    t0 = time.time()
    D = dict(A=dict(arms=arms), B=dict(arms=arms, arm_onehot=True), B_sub=dict(arms=arms, arm_onehot=True, arm_x_subtype=True),
             EXP=dict(arms=arms, exp=True, exp_x_subtype=True))

    def cv(design, **kw):
        return pd.concat(P(delayed(fit_one)(df, f, design, **kw) for f in FD), ignore_index=True)

    res = dict(n_transitions=n, n_patients=int(df.patient_id.nunique()), arms={k: int(v) for k, v in
                                                                                             df.drop_duplicates('patient_id').arm.value_counts().items()})
    oof, table = {}, []

    def log(name, o, cmp=None, mask=None, hgb=False):
        oof[name] = o
        row = dict(experiment=name, model='hgb' if hgb else 'enet', mae=float(np.abs(df.y.to_numpy() - avg(o, n)).mean()))
        if cmp:
            st = paired(df, o, oof[cmp], a.boot, mask); res.setdefault('comparisons', {})[name] = st
            row |= dict(vs=cmp, dmae=st['dmae'], ci_lo=st['ci'][0], ci_hi=st['ci'][1], p_boot=st['p_boot'], rel_pct=st['rel_pct'])
        table.append(row)
        print(f"[{time.time() - t0:6.0f}s] {name:22s} MAE {row['mae']:.4f}" + (f"  dMAE {row['dmae']:+.4f} [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] ({row['rel_pct']:+.1f}%)" if cmp else ''), flush=True)

    log('A', cv(D['A'])); log('B', cv(D['B']), 'A')
    # permutation null: arms permuted among patients within clinical subtype (randomization structure), training rows only
    pts = df.drop_duplicates('patient_id').set_index('patient_id')
    null = []
    for s in range(a.perm):
        rng = np.random.default_rng(s); amap = {}
        for sub, g in pts.groupby('subtype'):
            amap |= dict(zip(g.index, rng.permutation(g.arm.to_numpy())))
        null.append(paired(df, cv(D['B'], arm_map=amap), oof['A'], 200)['dmae'])
    null = np.array(null); st = res['comparisons']['B']
    thr = float(np.quantile(np.abs(null), .95))
    res['H1_permutation'] = dict(n=a.perm, null_mean=float(null.mean()), null_q05=float(np.quantile(null, .05)), abs_q95=thr,
                                 p_one_sided=float((1 + (null <= st['dmae']).sum()) / (1 + a.perm)), null=null.tolist())
    strong = st['ci'][1] < 0 and res['H1_permutation']['p_one_sided'] < .05 and st['repeats_negative'] >= 4
    res['H1_decision'] = 'strong' if strong else 'suggestive' if st['dmae'] < 0 and abs(st['dmae']) > thr else 'none'
    print('H1 decision', res['H1_decision'], 'perm p', res['H1_permutation']['p_one_sided'], flush=True)

    sec = []
    log('B_sub', cv(D['B_sub']), 'A'); sec.append('B_sub')
    log('EXP', cv(D['EXP']), 'A'); sec.append('EXP')
    for nm, m in (('B_regimen1', df.k.isin([0, 1]).to_numpy()), ('B_AC', (df.k == 2).to_numpy())):
        res['comparisons'][nm] = paired(df, oof['B'], oof['A'], a.boot, m); sec.append(nm)
        c = res['comparisons'][nm]; print(f"{nm:22s} dMAE {c['dmae']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] ({c['rel_pct']:+.1f}%) n={c['n_transitions']}", flush=True)
    log('A_hgb', cv(D['A'], hgb=True), hgb=True); log('B_hgb', cv(D['B'], hgb=True), 'A_hgb', hgb=True); sec.append('B_hgb')
    adj = holm([res['comparisons'][k]['p_boot'] for k in sec])
    res['secondary_holm'] = {k: dict(dmae=res['comparisons'][k]['dmae'], p=res['comparisons'][k]['p_boot'], p_holm=v) for k, v in zip(sec, adj)}

    # negative controls
    freq = pts.arm.to_numpy()
    fake = [paired(df, cv(D['B'], fake=dict(zip(pts.index, np.random.default_rng(5000 + s).permutation(freq)))), oof['A'], 200)['dmae']
            for s in range(a.perm)]
    dtp = [paired(df, cv(D['B'], dt_seed=7000 + s), oof['A'], 200)['dmae'] for s in range(a.perm)]
    res['negative_controls'] = {k: dict(mean=float(np.mean(v)), q05=float(np.quantile(v, .05)), frac_le_observed=float(np.mean(np.array(v) <= st['dmae'])))
                                for k, v in (('fake_arms', fake), ('dt_permutation', dtp))}

    # pCR from the twin state at T1 (+/- arm)
    clin = pd.read_excel(CLIN).assign(patient_id=lambda d: d.Patient_ID.astype(str)).set_index('patient_id')
    t01 = df[df.k == 0].set_index('patient_id')
    t12 = df[df.k == 1].set_index('patient_id')
    pc = t01[['log_ftv', 'log_ld', 'sphericity', 'bpe', 'HR', 'HER2', 'MP', 'age', 'postmeno', 'subtype', 'arm', 'exp', 'dt']].copy()
    pc['early_logratio'] = t01.y                                      # T0 -> T1 change: known at T1
    pc = pc.join(clin.pCR, how='inner').dropna(subset=['pCR'])
    pc = pc.loc[pc.index.isin(t12.index) | True]
    feats = ['log_ftv', 'log_ld', 'sphericity', 'bpe', 'HR', 'HER2', 'MP', 'age', 'postmeno', 'early_logratio']

    def pcr_cv(with_arm):
        preds = []
        for r in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(pc, pc.arm + pc.pCR.astype(str)):
                Xtr, Xte = pc.iloc[tr], pc.iloc[te]

                def mk(X, med):
                    Z = X[feats].fillna(med).to_numpy(float)
                    if with_arm:
                        Z = np.c_[Z, np.column_stack([(X.arm == a_).to_numpy(float) for a_ in arms])]
                    return Z
                med = Xtr[feats].median()
                gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=.5, max_iter=20000)),
                                  {'logisticregression__C': [.01, .03, .1, .3, 1]}, cv=3, scoring='roc_auc').fit(mk(Xtr, med), Xtr.pCR)
                preds.append(pd.DataFrame(dict(pid=pc.index[te], p=gs.predict_proba(mk(Xte, med))[:, 1])))
        return pd.concat(preds).groupby('pid').p.mean().loc[pc.index].to_numpy()
    p0, p1 = pcr_cv(False), pcr_cv(True); yy = pc.pCR.to_numpy().astype(int)
    rng = np.random.default_rng(3); bs = []
    for _ in range(a.boot):
        i = rng.integers(0, len(yy), len(yy))
        if yy[i].min() != yy[i].max():
            bs.append(roc_auc_score(yy[i], p1[i]) - roc_auc_score(yy[i], p0[i]))

    def slope(p):
        lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        return float(LogisticRegression(C=1e6).fit(lp[:, None], yy).coef_[0, 0])
    res['pcr'] = dict(n=len(yy), prevalence=float(yy.mean()), auroc_state=float(roc_auc_score(yy, p0)), auroc_state_arm=float(roc_auc_score(yy, p1)),
                      d_auroc=float(roc_auc_score(yy, p1) - roc_auc_score(yy, p0)), d_auroc_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                      brier_state=float(brier_score_loss(yy, p0)), brier_state_arm=float(brier_score_loss(yy, p1)), slope_state=slope(p0), slope_state_arm=slope(p1))
    print('pCR', {k: v for k, v in res['pcr'].items()}, flush=True)

    for k, o in oof.items():
        o.assign(analysis=k).to_csv(run / f'oof_{k}.csv', index=False)
    pd.DataFrame(dict(patient_id=pc.index, pcr=yy, p_state=p0, p_state_arm=p1)).to_csv(run / 'oof_pcr.csv', index=False)
    pd.DataFrame(table).to_csv(run / 'experiment_table.csv', index=False)
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    print(f'done {time.time() - t0:.0f}s -> {run}')


if __name__ == '__main__':
    main()
