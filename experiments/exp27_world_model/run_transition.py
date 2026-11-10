#!/usr/bin/env python3
"""Treatment-conditioned state transition S_FU = f(S_BL, A, dt), as pre-specified in the analysis plan.

Uses the locked splits, so every comparison is paired on identical folds. All imputation / scaling / spline knots / centring / tuning happen inside the
outer training fold (tuning on the locked inner folds). Writes outputs/exp27/transition/: oof.csv (every out-of-fold
prediction), results.json, experiment_table.csv.

  python run_transition.py [--perm 200] [--boot 5000] [--jobs 96] [--quick]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import BayesianRidge, ElasticNet, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import state  # noqa: E402

REPO = state.REPO
OUT = REPO / 'outputs/exp27/transition'
S = list(state.S_FEATURES)
A, DT = state.ACTION_FEATURE, state.HORIZON_FEATURE
EXTENT = 'extent_multi_or_gt50'                        # multinodular OR involvement > 50 % (bilobar / extensive disease proxy)
H1_INTER = ['child_pugh_b_or_c', EXTENT, 'log_seg_mass_ml', 'bl_mrecist_mm']


# ------------------------------------------------------------------------------------------------- data
def load():
    _, fr = state.build()
    tmp = OUT / '_aux_audit'; tmp.mkdir(parents=True, exist_ok=True)
    tm = pd.read_csv(state.AUDIT_RUN / 'transition_manifest.csv')
    aux_ids = tm.loc[tm.exclusion_reason == 'treatment unknown', 'patient_id'].tolist()
    tm2 = tm.copy(); tm2['transition_eligible'] = tm2.patient_id.isin(aux_ids)
    tm2.to_csv(tmp / 'transition_manifest.csv', index=False)
    (tmp / 'segmentation_manifest.csv').write_text((state.AUDIT_RUN / 'segmentation_manifest.csv').read_text())
    _, aux = state.build(tmp)
    aux[A] = np.nan                                                    # regimen unknown: never used for a treatment term
    for d in (fr, aux):
        d[EXTENT] = ((d.multinodular == 1) | (d.involvement_gt50 == 1)).astype(float)
        d['y_abs_mm'] = d.y_fu_mm - d.bl_mrecist_mm
        d['y_logratio'] = np.log((d.y_fu_mm + 1) / (d.bl_mrecist_mm + 1))
    return fr, aux


# ------------------------------------------------------------------------------------------------- features (fitted in-fold)
def rcs_basis(x, knots):
    """Restricted (natural) cubic spline, Harrell: x plus len(knots) - 2 non-linear terms."""
    k = np.asarray(knots); x = np.asarray(x, float)
    cols = [x]
    d = (k[-1] - k[-2])
    for j in range(len(k) - 2):
        t = lambda z: np.maximum(z, 0) ** 3  # noqa: E731
        cols.append((t(x - k[j]) - t(x - k[-2]) * (k[-1] - k[j]) / d + t(x - k[-1]) * (k[-2] - k[j]) / d) / (k[-1] - k[0]) ** 2)
    return np.column_stack(cols)


class Design(BaseEstimator, TransformerMixin):
    """State features (+ missing indicators), optional treatment, horizon representation and A x feature interactions.
    Medians, knots and centring constants are learned on the data passed to fit (the outer / inner training fold)."""

    def __init__(self, treat=False, horizon=None, inter=(), horizon_x_treat=False):
        self.treat, self.horizon, self.inter, self.horizon_x_treat = treat, horizon, inter, horizon_x_treat

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.med_ = X[S].median()
        self.miss_ = [c for c in S if X[c].isna().any()]
        self.cent_ = {c: X[c].fillna(self.med_.get(c, 0)).mean() for c in self.inter}
        if self.horizon in ('rcs', 'rcs_x_a'):
            self.knots_ = np.quantile(X[DT], [.05, .35, .65, .95])
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        Z = X[S].fillna(self.med_)
        parts = [Z.to_numpy(), X[self.miss_].isna().to_numpy(float)]
        a = X[A].to_numpy(float) if self.treat else None
        if self.treat:
            parts.append(a[:, None])
            for c in self.inter:
                parts.append((a * (X[c].fillna(self.med_.get(c, 0)).to_numpy(float) - self.cent_[c]))[:, None])
        dt = X[DT].to_numpy(float)
        if self.horizon == 'linear':
            parts.append(dt[:, None])
        elif self.horizon == 'log':
            parts.append(np.log(dt)[:, None])
        elif self.horizon in ('rcs', 'rcs_x_a'):
            B = rcs_basis(dt, self.knots_); parts.append(B)
            if self.horizon == 'rcs_x_a' and self.treat:
                parts.append(B * a[:, None])
        elif self.horizon == 'piecewise':
            parts.append(np.column_stack([dt, np.maximum(dt - 8, 0), np.maximum(dt - 16, 0)]))
        return np.column_stack(parts)


SPEC = {   # name -> Design kwargs
    'state': dict(),
    'H1': dict(treat=True, horizon='rcs', inter=tuple(H1_INTER)),
    'S+A': dict(treat=True),
    'S+A+lin': dict(treat=True, horizon='linear'),
    'S+A+log': dict(treat=True, horizon='log'),
    'S+A+rcs': dict(treat=True, horizon='rcs'),
    'S+A+piecewise': dict(treat=True, horizon='piecewise'),
    'S+A+rcs+Axrcs': dict(treat=True, horizon='rcs_x_a'),
    'S+rcs': dict(horizon='rcs'),                       # post-hoc decomposition (horizon without treatment), descriptive only
}


def learner(model):
    if model == 'enet':
        return ElasticNet(max_iter=50000), {'alpha': [0.1, 0.3, 1, 3, 10, 30], 'l1_ratio': [0.05, 0.5, 0.95]}
    if model == 'gbr':
        return GradientBoostingRegressor(loss='huber', subsample=0.8, min_samples_leaf=5, learning_rate=0.05, random_state=0), {
            'max_depth': [1, 2], 'n_estimators': [50, 150]}
    if model == 'rf':
        return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=1), {'min_samples_leaf': [3, 8], 'max_features': [.33, 1.0]}
    if model == 'et':
        return ExtraTreesRegressor(n_estimators=300, random_state=0, n_jobs=1), {'min_samples_leaf': [3, 8], 'max_features': [.33, 1.0]}
    if model == 'hgb':
        return HistGradientBoostingRegressor(loss='absolute_error', random_state=0), {
            'max_depth': [2, 3], 'learning_rate': [.05], 'max_iter': [50, 150], 'min_samples_leaf': [8]}
    if model == 'bayes':
        return BayesianRidge(), {'alpha_1': [1e-6, 1e-3]}
    raise ValueError(model)


def pipe(spec, model):
    est, grid = learner(model)
    steps = [Design(**SPEC[spec])]
    if model in ('enet', 'bayes'):
        steps.append(StandardScaler())
    p = make_pipeline(*steps, est)
    name = p.steps[-1][0]
    return p, {f'{name}__{k}': v for k, v in grid.items()}


def inner_cv(fold, tr):
    pos = {p: i for i, p in enumerate(tr)}
    return [(np.array([pos[p] for p in f['train'] if p in pos]), np.array([pos[p] for p in f['valid'] if p in pos])) for f in fold['inner']]


def fit_predict(fr, fold, spec, model, target='y_pct', a_perm_seed=None, fake_seed=None, dt_perm_seed=None, sub=None, aux=None):
    """One outer fold. Returns OOF predictions for the fold's test patients."""
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    tr = [p for p in fold['train'] if pd.notna(fr.at[p, target])]
    if sub is not None:
        frac, seed = sub
        rng = np.random.default_rng(seed)
        tr = sorted(rng.choice(tr, max(int(round(frac * len(tr))), 12), replace=False).tolist())
    te = [p for p in fold['test'] if pd.notna(fr.at[p, target])]
    X_tr, X_te = fr.loc[tr].copy(), fr.loc[te].copy()
    if fake_seed is not None:                                         # random fake treatment, same prevalence, train and test
        rng = np.random.default_rng(10_000 + fake_seed)
        X_tr[A] = rng.binomial(1, 30 / 82, len(X_tr))
        X_te[A] = rng.binomial(1, 30 / 82, len(X_te))
    if a_perm_seed is not None:                                       # treatment permuted among training patients only
        X_tr[A] = np.random.default_rng(a_perm_seed).permutation(X_tr[A].to_numpy())
    if dt_perm_seed is not None:
        X_tr[DT] = np.random.default_rng(20_000 + dt_perm_seed).permutation(X_tr[DT].to_numpy())
    p, grid = pipe(spec, model)
    cv = inner_cv(fold, tr) if sub is None else 3
    gs = GridSearchCV(p, grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=1, error_score='raise')
    gs.fit(X_tr, fr.loc[tr, target].to_numpy())
    return pd.DataFrame(dict(patient_id=te, repeat=fold['repeat'], fold=fold['fold'], y_true=fr.loc[te, target].to_numpy(),
                             y_pred=gs.predict(X_te)))


def fit_two_stage(fr, aux, fold, with_treat):
    """H2: stage 1 state-only GBR on primary training + auxiliary rows (tuned on inner folds; auxiliary rows always in inner-train);
    stage 2 Elastic Net on the stage-1 residual with horizon spline (+ H1 treatment terms), primary training rows only."""
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    tr, te = fold['train'], fold['test']
    Xall = pd.concat([fr.loc[tr], aux]); yall = Xall.y_pct.to_numpy()
    n = len(tr); aux_idx = np.arange(n, len(Xall))
    pos = {p: i for i, p in enumerate(tr)}
    cv = [(np.r_[np.array([pos[p] for p in f['train']]), aux_idx], np.array([pos[p] for p in f['valid']])) for f in fold['inner']]
    p1, g1 = pipe('state', 'gbr')
    gs1 = GridSearchCV(p1, g1, cv=cv, scoring='neg_mean_absolute_error', n_jobs=1).fit(Xall, yall)
    s1_tr = np.zeros(n)                                               # out-of-fold stage-1 predictions for the training rows
    for a_, b_ in cv:
        m = clone(gs1.best_estimator_).fit(Xall.iloc[a_], yall[a_]); s1_tr[b_] = m.predict(Xall.iloc[b_])
    s1_te = gs1.best_estimator_.predict(fr.loc[te])
    kw = dict(horizon='rcs') | (dict(treat=True, inter=tuple(H1_INTER)) if with_treat else {})
    p2 = make_pipeline(Design(**kw), StandardScaler(), ElasticNet(max_iter=50000))
    g2 = {'elasticnet__alpha': [0.3, 1, 3, 10, 30], 'elasticnet__l1_ratio': [0.05, 0.5, 0.95]}
    r_tr = fr.loc[tr, 'y_pct'].to_numpy() - s1_tr
    gs2 = GridSearchCV(p2, g2, cv=inner_cv(fold, tr), scoring='neg_mean_absolute_error', n_jobs=1).fit(fr.loc[tr], r_tr)
    return pd.DataFrame(dict(patient_id=te, repeat=fold['repeat'], fold=fold['fold'], y_true=fr.loc[te, 'y_pct'].to_numpy(),
                             y_pred=s1_te + gs2.predict(fr.loc[te])))


# ------------------------------------------------------------------------------------------------- statistics
def per_patient(df):
    return df.groupby('patient_id').agg(y_true=('y_true', 'first'), y_pred=('y_pred', 'mean'))


def mae(df):
    d = per_patient(df); return float((d.y_true - d.y_pred).abs().mean())


def paired(df_a, df_b, n_boot, seed=0):
    """dMAE = MAE(a) - MAE(b) on repeat-averaged OOF predictions; patient-level paired bootstrap; per-repeat signs."""
    a, b = per_patient(df_a), per_patient(df_b).loc[per_patient(df_a).index]
    ea, eb = (a.y_true - a.y_pred).abs().to_numpy(), (b.y_true - b.y_pred).abs().to_numpy()
    d = ea - eb
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    reps = []
    for r in sorted(df_a.repeat.unique()):
        ra, rb = df_a[df_a.repeat == r].set_index('patient_id'), df_b[df_b.repeat == r].set_index('patient_id')
        reps.append(float((ra.y_true - ra.y_pred).abs().mean() - (rb.y_true - rb.y_pred).abs().loc[ra.index].mean()))
    p2 = float(min(1, 2 * min((bs >= 0).mean(), (bs <= 0).mean())))
    return dict(mae_a=float(ea.mean()), mae_b=float(eb.mean()), dmae=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                p_boot_two_sided=p2, repeats_negative=int(sum(x < 0 for x in reps)), per_repeat=reps,
                medae_a=float(np.median(ea)), medae_b=float(np.median(eb)),
                rmse_a=float(np.sqrt(((a.y_true - a.y_pred) ** 2).mean())), rmse_b=float(np.sqrt(((b.y_true - b.y_pred) ** 2).mean())),
                spearman_a=float(a.y_true.corr(a.y_pred, method='spearman')), spearman_b=float(b.y_true.corr(b.y_pred, method='spearman')),
                r2_a=float(1 - ((a.y_true - a.y_pred) ** 2).sum() / ((a.y_true - a.y_true.mean()) ** 2).sum()),
                r2_b=float(1 - ((b.y_true - b.y_pred) ** 2).sum() / ((b.y_true - b.y_true.mean()) ** 2).sum()))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps); run = 0
    for i, k in enumerate(order):
        run = max(run, min(1, (m - i) * ps[k])); adj[k] = run
    return adj.tolist()


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perm', type=int, default=200); ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--jobs', type=int, default=96); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.quick:
        a.perm, a.boot = 5, 500
    fr, aux = load()
    folds = json.loads((REPO / 'outputs/exp27/splits/splits.json').read_text())['folds']
    run = OUT; run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = Parallel(n_jobs=a.jobs, verbose=0)

    def cv(spec, model, target='y_pct', **kw):
        return pd.concat(P(delayed(fit_predict)(fr, f, spec, model, target, **kw) for f in folds), ignore_index=True)

    oof, res, table = {}, {'n_primary': len(fr), 'n_aux': len(aux), 'arms': fr[A].value_counts().to_dict()}, []

    def log(name, spec, model, target, df, cmp=None):
        oof[name] = df.assign(analysis=name)
        row = dict(experiment=name, features=spec, model=model, target=target, cv='5x5 nested', n=int(df.patient_id.nunique()),
                   mae=mae(df), spearman=float(per_patient(df).y_true.corr(per_patient(df).y_pred, method='spearman')))
        if cmp is not None:
            st = paired(df, oof[cmp], a.boot); row |= dict(vs=cmp, dmae=st['dmae'], ci_lo=st['ci'][0], ci_hi=st['ci'][1],
                                                          p_boot=st['p_boot_two_sided'], repeats_negative=st['repeats_negative'])
            res.setdefault('comparisons', {})[name] = st
        table.append(row)
        print(f"[{time.time() - t0:6.0f}s] {name:28s} MAE {row['mae']:.2f}" + (f"  dMAE {row['dmae']:+.2f} [{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]" if cmp else ''),
              flush=True)

    # ---- primary H1 (Elastic Net)
    log('state|enet', 'state', 'enet', 'y_pct', cv('state', 'enet'))
    log('H1|enet', 'H1', 'enet', 'y_pct', cv('H1', 'enet'), cmp='state|enet')
    perm = np.array([paired(cv('H1', 'enet', a_perm_seed=s), oof['state|enet'], 200)['dmae'] for s in range(a.perm)])
    obs = res['comparisons']['H1|enet']['dmae']
    res['H1_permutation'] = dict(n=a.perm, null_mean=float(perm.mean()), null_q05=float(np.quantile(perm, .05)),
                                 p_one_sided=float((1 + (perm <= obs).sum()) / (1 + a.perm)), null=perm.tolist())
    st = res['comparisons']['H1|enet']
    strong = st['ci'][1] < 0 and res['H1_permutation']['p_one_sided'] < .05 and st['repeats_negative'] >= 4
    res['H1_decision'] = 'strong' if strong else 'suggestive' if st['dmae'] < 0 else 'none'
    print('H1 decision:', res['H1_decision'], 'perm p', res['H1_permutation']['p_one_sided'], flush=True)

    # ---- H2 two-stage with auxiliary patients
    for wt in (False, True):
        df = pd.concat(P(delayed(fit_two_stage)(fr, aux, f, wt) for f in folds), ignore_index=True)
        log(f"H2|{'treat' if wt else 'no_treat'}", 'two-stage', 'gbr+enet', 'y_pct', df, cmp='H2|no_treat' if wt else None)
    st = res['comparisons']['H2|treat']
    res['H2_decision'] = 'strong' if st['ci'][1] < 0 and st['repeats_negative'] >= 4 else 'suggestive' if st['dmae'] < 0 else 'none'

    # ---- secondary 1: horizon representations (Elastic Net)
    sec = []
    for spec in ('S+A', 'S+A+lin', 'S+A+log', 'S+A+rcs', 'S+A+piecewise', 'S+A+rcs+Axrcs'):
        log(f'{spec}|enet', spec, 'enet', 'y_pct', cv(spec, 'enet'), cmp='state|enet'); sec.append(f'{spec}|enet')
    log('S+rcs|enet', 'S+rcs', 'enet', 'y_pct', cv('S+rcs', 'enet'), cmp='state|enet')          # post-hoc decomposition
    log('H1_vs_S+rcs|enet', 'H1', 'enet', 'y_pct', oof['H1|enet'].drop(columns='analysis'), cmp='S+rcs|enet')

    # ---- secondary 2: model families
    for m in ('gbr', 'rf', 'et', 'hgb', 'bayes'):
        log(f'state|{m}', 'state', m, 'y_pct', cv('state', m))
        log(f'H1|{m}', 'H1', m, 'y_pct', cv('H1', m), cmp=f'state|{m}'); sec.append(f'H1|{m}')

    # ---- secondary 4: outcome representations
    fr['y_wins'] = fr.y_pct                                            # winsorised inside folds below
    for tgt in ('y_abs_mm', 'y_logratio', 'y_fu_mm'):
        log(f'state|enet|{tgt}', 'state', 'enet', tgt, cv('state', 'enet', tgt))
        log(f'H1|enet|{tgt}', 'H1', 'enet', tgt, cv('H1', 'enet', tgt), cmp=f'state|enet|{tgt}'); sec.append(f'H1|enet|{tgt}')
    adj = holm([res['comparisons'][k]['p_boot_two_sided'] for k in sec])
    res['secondary_holm'] = {k: dict(p=res['comparisons'][k]['p_boot_two_sided'], p_holm=v, dmae=res['comparisons'][k]['dmae'])
                             for k, v in zip(sec, adj)}

    # ---- ORR classification (secondary)
    def orr_cv(spec):
        rows = []
        for f in folds:
            tr, te = f['train'], f['test']
            p = make_pipeline(Design(**SPEC[spec]), StandardScaler(), LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=.5,
                                                                                          max_iter=20000))
            gs = GridSearchCV(p, {'logisticregression__C': [.01, .03, .1, .3, 1]}, cv=inner_cv(f, tr), scoring='roc_auc')
            gs.fit(fr.loc[tr], fr.loc[tr, 'orr'])
            rows.append(pd.DataFrame(dict(patient_id=te, repeat=f['repeat'], y_true=fr.loc[te, 'orr'].to_numpy(), y_pred=gs.predict_proba(fr.loc[te])[:, 1])))
        d = per_patient(pd.concat(rows))
        lp = np.log(d.y_pred.clip(1e-4, 1 - 1e-4) / (1 - d.y_pred.clip(1e-4, 1 - 1e-4)))
        slope = LogisticRegression(C=1e6).fit(lp.to_numpy()[:, None], d.y_true).coef_[0, 0]
        return dict(auroc=float(roc_auc_score(d.y_true, d.y_pred)), brier=float(brier_score_loss(d.y_true, d.y_pred)), cal_slope=float(slope))
    res['orr'] = {s: orr_cv(s) for s in ('state', 'H1')}

    # ---- secondary 3: meta-learners (conditional response difference; NOT individual causal effects)
    def meta(f):
        tr, te = f['train'], f['test']
        out = {}
        g = lambda: GradientBoostingRegressor(loss='huber', max_depth=2, n_estimators=100, learning_rate=.05, subsample=.8,  # noqa: E731
                                              min_samples_leaf=5, random_state=0)
        Xtr = Design(horizon='rcs').fit(fr.loc[tr]); Ztr, Zte = Xtr.transform(fr.loc[tr]), Xtr.transform(fr.loc[te])
        atr = fr.loc[tr, A].to_numpy(); ytr = fr.loc[tr, 'y_pct'].to_numpy()
        sl = g().fit(np.c_[Ztr, atr], ytr)
        out['S'] = sl.predict(np.c_[Zte, np.ones(len(te))]) - sl.predict(np.c_[Zte, np.zeros(len(te))])
        m1, m0 = g().fit(Ztr[atr == 1], ytr[atr == 1]), g().fit(Ztr[atr == 0], ytr[atr == 0])
        out['T'] = m1.predict(Zte) - m0.predict(Zte)
        ps = LogisticRegression(C=.3, max_iter=5000).fit(StandardScaler().fit(Ztr).transform(Ztr), atr)
        e = ps.predict_proba(StandardScaler().fit(Ztr).transform(Zte))[:, 1]
        d1 = ytr[atr == 1] - m0.predict(Ztr[atr == 1]); d0 = m1.predict(Ztr[atr == 0]) - ytr[atr == 0]
        t1, t0_ = g().fit(Ztr[atr == 1], d1), g().fit(Ztr[atr == 0], d0)
        out['X'] = e * t0_.predict(Zte) + (1 - e) * t1.predict(Zte)
        tpred = np.where(fr.loc[te, A] == 1, m1.predict(Zte), m0.predict(Zte))
        return pd.DataFrame(dict(patient_id=te, repeat=f['repeat'], cate_S=out['S'], cate_T=out['T'], cate_X=out['X'],
                                 y_true=fr.loc[te, 'y_pct'].to_numpy(), y_pred=tpred))
    mt = pd.concat(P(delayed(meta)(f) for f in folds), ignore_index=True)
    oof['meta'] = mt.assign(analysis='meta')
    log('T-learner|gbr', 'state+rcs per arm', 'gbr', 'y_pct', mt[['patient_id', 'repeat', 'y_true', 'y_pred']].assign(fold=0), cmp='state|gbr')
    rep_corr = {k: float(np.nanmean([mt[mt.repeat == r].set_index('patient_id')[k].corr(mt[mt.repeat == r2].set_index('patient_id')[k])
                                     for r in range(5) for r2 in range(5) if r < r2])) for k in ('cate_S', 'cate_T', 'cate_X')}
    res['meta'] = dict(cross_repeat_correlation=rep_corr,
                       mean=mt.groupby('patient_id')[['cate_S', 'cate_T', 'cate_X']].mean().mean().to_dict())

    # ---- secondary 5: propensity, overlap, weighted dMAE, AIPW
    ps_rows = []
    for f in folds:
        tr, te = f['train'], f['test']
        p = make_pipeline(Design(), StandardScaler(), LogisticRegression(C=.3, max_iter=5000)).fit(fr.loc[tr], fr.loc[tr, A])
        ps_rows.append(pd.DataFrame(dict(patient_id=te, repeat=f['repeat'], e=p.predict_proba(fr.loc[te])[:, 1])))
    e = pd.concat(ps_rows).groupby('patient_id').e.mean().loc[fr.index]
    oof['propensity'] = e.rename('y_pred').reset_index().assign(analysis='propensity')
    w_ov = e * (1 - e); w_ipt = np.where(fr[A] == 1, 1 / e, 1 / (1 - e))
    ess = lambda w: float(w.sum() ** 2 / (w ** 2).sum())  # noqa: E731
    h1, s0 = per_patient(oof['H1|enet']).loc[fr.index], per_patient(oof['state|enet']).loc[fr.index]
    d = ((h1.y_true - h1.y_pred).abs() - (s0.y_true - s0.y_pred).abs()).to_numpy()
    # AIPW with cross-fitted T-learner arm models (re-predicted per arm)
    def arms(f):
        tr, te = f['train'], f['test']
        D = Design(horizon='rcs').fit(fr.loc[tr]); Ztr, Zte = D.transform(fr.loc[tr]), D.transform(fr.loc[te])
        atr, ytr = fr.loc[tr, A].to_numpy(), fr.loc[tr, 'y_pct'].to_numpy()
        g = lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=3, l1_ratio=.5, max_iter=50000))  # noqa: E731
        return pd.DataFrame(dict(patient_id=te, mu1=g().fit(Ztr[atr == 1], ytr[atr == 1]).predict(Zte),
                                 mu0=g().fit(Ztr[atr == 0], ytr[atr == 0]).predict(Zte)))
    M = pd.concat(P(delayed(arms)(f) for f in folds)).groupby('patient_id').mean().loc[fr.index]
    y, t, ee = fr.y_pct.to_numpy(), fr[A].to_numpy(), e.clip(.02, .98).to_numpy()
    phi = M.mu1 - M.mu0 + t * (y - M.mu1) / ee - (1 - t) * (y - M.mu0) / (1 - ee)
    keep = (e >= .05) & (e <= .95)
    rng = np.random.default_rng(1)
    bs = [phi[keep].to_numpy()[rng.integers(0, keep.sum(), keep.sum())].mean() for _ in range(a.boot)]
    res['propensity'] = dict(range=[float(e.min()), float(e.max())], outside_005_095=int((~keep).sum()),
                             ess_overlap=ess(w_ov), ess_iptw=ess(pd.Series(w_ipt)),
                             dmae_overlap_weighted=float(np.average(d, weights=w_ov)), dmae_iptw=float(np.average(d, weights=w_ipt)),
                             aipw_ate_pct_points=float(phi[keep].mean()), aipw_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                             smd={c: float((fr.loc[t == 1, c].mean() - fr.loc[t == 0, c].mean()) /
                                           np.sqrt((fr.loc[t == 1, c].var() + fr.loc[t == 0, c].var()) / 2)) for c in S + [DT]})

    # ---- negative controls (H1 Elastic Net)
    nc = {}
    nc['fake_treatment'] = [paired(cv('H1', 'enet', fake_seed=s), oof['state|enet'], 200)['dmae'] for s in range(a.perm)]
    nc['horizon_permutation'] = [paired(cv('H1', 'enet', dt_perm_seed=s), oof['state|enet'], 200)['dmae'] for s in range(a.perm)]
    res['negative_controls'] = {k: dict(mean=float(np.mean(v)), q05=float(np.quantile(v, .05)), q95=float(np.quantile(v, .95)),
                                        frac_below_observed=float(np.mean(np.array(v) <= obs)), values=v) for k, v in nc.items()}
    res['negative_controls']['treatment_permutation'] = {k: v for k, v in res['H1_permutation'].items() if k != 'null'}

    # ---- learning curve
    lc = []
    for frac in (.3, .5, .7, 1.0):
        for spec in ('state', 'H1'):
            ms = [mae(cv(spec, 'enet', sub=(frac, 100 * i + int(frac * 10)))) for i in range(5 if frac < 1 else 1)]
            lc.append(dict(frac=frac, spec=spec, mae_mean=float(np.mean(ms)), mae_sd=float(np.std(ms))))
    res['learning_curve'] = lc

    # ---- error analysis
    ea = pd.DataFrame(dict(d_abs_err=d), index=fr.index).join(fr[[A, DT, 'bl_mrecist_mm', 'log_seg_mass_ml', 'child_pugh_b_or_c', EXTENT,
                                                                    'log_afp', 'bclc_stage', 'y_pct']])
    res['error_analysis'] = dict(improved=ea.nsmallest(10, 'd_abs_err').reset_index().to_dict('records'),
                                 worsened=ea.nlargest(10, 'd_abs_err').reset_index().to_dict('records'))

    pd.concat(oof.values(), ignore_index=True).to_csv(run / 'oof.csv', index=False)
    pd.DataFrame(table).to_csv(run / 'experiment_table.csv', index=False)
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    print(f'done in {time.time() - t0:.0f}s -> {run}')


if __name__ == '__main__':
    main()
