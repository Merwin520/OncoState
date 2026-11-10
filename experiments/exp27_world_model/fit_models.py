"""Experiment 2.7 Phases 8-11 (+ shuffle control, reader sensitivity): out-of-fold predictions on the locked splits.

Transition: S_BL + A_TACE + dt_BL->FU -> S_FU. Primary target: mRECIST % change from baseline (median of readers).
Variants:  WM-0 = S ; WM-1 = S + A ; WM-2 = S + dt ; WM-3 = S + A + dt ;
           WM-shuffle = S + A' + dt, A' permuted among TRAINING patients only (test rows keep their true A; targets never shuffled).
Naive:     training mean, training median, no change (0 %), training treatment-arm mean.
Models:    Elastic Net and gradient boosting (required), small MLP (exploratory). Direct ORR classifiers (secondary).
All imputation / scaling / tuning happens inside sklearn pipelines fitted on the outer training fold; tuning uses the
locked inner folds. Writes outputs/exp27/models/run/.
"""
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import state

REPO = state.REPO
SPLITS = REPO / 'outputs/exp27/splits'
S = list(state.S_FEATURES)
VARIANTS = {'WM-0': S, 'WM-1': S + [state.ACTION_FEATURE], 'WM-2': S + [state.HORIZON_FEATURE],
            'WM-3': S + [state.ACTION_FEATURE, state.HORIZON_FEATURE]}
N_SHUFFLE = 20


def regressor(name):
    imp = SimpleImputer(strategy='median', add_indicator=True)
    if name == 'enet':
        return make_pipeline(imp, StandardScaler(), ElasticNet(max_iter=50000)), {
            'elasticnet__alpha': [0.1, 0.3, 1, 3, 10, 30], 'elasticnet__l1_ratio': [0.05, 0.5, 0.95]}
    if name == 'gbr':
        return make_pipeline(imp, GradientBoostingRegressor(loss='huber', subsample=0.8, min_samples_leaf=5, learning_rate=0.05, random_state=0)), {
            'gradientboostingregressor__max_depth': [1, 2], 'gradientboostingregressor__n_estimators': [50, 150]}
    if name == 'mlp':  # exploratory
        mlp = TransformedTargetRegressor(MLPRegressor(hidden_layer_sizes=(8,), max_iter=3000, random_state=0), transformer=StandardScaler())
        return make_pipeline(imp, StandardScaler(), mlp), {'transformedtargetregressor__regressor__alpha': [1.0, 10.0]}
    raise ValueError(name)


def classifier(name):
    imp = SimpleImputer(strategy='median', add_indicator=True)
    if name == 'logit':
        return make_pipeline(imp, StandardScaler(), LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=20000)), {
            'logisticregression__C': [0.01, 0.03, 0.1, 0.3, 1.0]}
    if name == 'gbc':
        return make_pipeline(imp, GradientBoostingClassifier(subsample=0.8, min_samples_leaf=5, learning_rate=0.05, random_state=0)), {
            'gradientboostingclassifier__max_depth': [1, 2], 'gradientboostingclassifier__n_estimators': [50, 150]}
    raise ValueError(name)


def run_task(fr, fold, task):
    """One outer fold x model x variant (x shuffle seed / target). Returns OOF rows for the test patients."""
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    t0 = time.time()
    tgt, kind = task['target'], task['kind']
    tr = [p for p in fold['train'] if pd.notna(fr.at[p, tgt])]
    te = [p for p in fold['test'] if pd.notna(fr.at[p, tgt])]
    y_tr = fr.loc[tr, tgt].to_numpy()
    base = dict(task_id=task['id'], target=tgt, kind=kind, model=task['model'], variant=task['variant'], variant_label=task['variant_label'],
                shuffle_seed=task.get('shuffle_seed', -1), repeat=fold['repeat'], fold=fold['fold'])
    if kind == 'naive':
        m = task['model']
        if m == 'train_mean':
            pred = np.full(len(te), y_tr.mean())
        elif m == 'train_median':
            pred = np.full(len(te), np.median(y_tr))
        elif m == 'no_change':
            pred = np.zeros(len(te))
        elif m == 'train_arm_mean':
            arm = fr.loc[tr].groupby('deb_tace')[tgt].mean()
            pred = fr.loc[te, 'deb_tace'].map(arm).to_numpy()
        params = {}
    else:
        cols = task.get('cols') or VARIANTS[task['variant']]  # explicit feature list (ablation) or a named variant
        X_tr, X_te = fr.loc[tr, cols].copy(), fr.loc[te, cols]
        if task['variant_label'] == 'WM-shuffle':
            X_tr[state.ACTION_FEATURE] = np.random.default_rng(task['shuffle_seed']).permutation(X_tr[state.ACTION_FEATURE].to_numpy())
        pos = {p: i for i, p in enumerate(tr)}
        inner = [(np.array([pos[p] for p in f['train'] if p in pos]), np.array([pos[p] for p in f['valid'] if p in pos])) for f in fold['inner']]
        if kind == 'reg':
            est, grid = regressor(task['model'])
            gs = GridSearchCV(est, grid, cv=inner, scoring='neg_mean_absolute_error', n_jobs=1, error_score='raise').fit(X_tr, y_tr)
            pred = gs.predict(X_te)
        else:
            est, grid = classifier(task['model'])
            gs = GridSearchCV(est, grid, cv=inner, scoring='roc_auc', n_jobs=1, error_score='raise').fit(X_tr, y_tr)
            pred = gs.predict_proba(X_te)[:, 1]
        params = gs.best_params_
    return [dict(base, patient_id=p, y_true=fr.at[p, tgt], y_pred=float(v), arm=int(fr.at[p, 'deb_tace']),
                 params=json.dumps(params), seconds=round(time.time() - t0, 3)) for p, v in zip(te, pred)]


def tasks():
    out = []
    for m in ('train_mean', 'train_median', 'no_change', 'train_arm_mean'):
        out.append(dict(kind='naive', target='y_pct', model=m, variant='naive', variant_label='naive'))
    for m in ('enet', 'gbr', 'mlp'):
        for v in VARIANTS:
            out.append(dict(kind='reg', target='y_pct', model=m, variant=v, variant_label=v))
    for m in ('enet', 'gbr'):
        for s in range(N_SHUFFLE):
            out.append(dict(kind='reg', target='y_pct', model=m, variant='WM-3', variant_label='WM-shuffle', shuffle_seed=s))
    for m in ('logit', 'gbc'):
        for v in VARIANTS:
            out.append(dict(kind='clf', target='orr', model=m, variant=v, variant_label=v))
    for rd in (1, 2, 3):
        for m in ('enet', 'gbr'):
            for v in ('WM-0', 'WM-3'):
                out.append(dict(kind='reg', target=f'y_pct_r{rd}', model=m, variant=v, variant_label=v))
        out.append(dict(kind='naive', target=f'y_pct_r{rd}', model='train_mean', variant='naive', variant_label='naive'))
    for i, t in enumerate(out):
        t['id'] = i
    return out


def main():
    sp = json.loads((SPLITS / 'splits.json').read_text())
    blob = (SPLITS / 'splits.json').read_text()
    if hashlib.sha256(blob.encode()).hexdigest() != (SPLITS / 'splits.sha256').read_text().strip():
        sys.exit('splits.json does not match its locked sha256')
    if not sp['audit_passed']:
        sys.exit(f"split audit failed: {sp['audit_failures']}")
    states, fr = state.build()
    if sorted(fr.index) != sorted(sp['cohort']):
        sys.exit('cohort differs from the locked split cohort')
    run = REPO / 'outputs/exp27/models' / os.environ.get('EXP27_RUN_NAME', 'run')
    run.mkdir(parents=True)
    state.write_states(states, run / 'patient_states')
    leak = state.check_no_leakage()
    for v, cols in VARIANTS.items():  # the only non-state inputs are A and dt; never a target or outcome
        extra = set(cols) - set(S)
        assert extra <= {state.ACTION_FEATURE, state.HORIZON_FEATURE}, (v, extra)
        assert not set(cols) & {'y_pct', 'y_fu_mm', 'orr', 'consensus_category', 'y_pct_r1', 'y_pct_r2', 'y_pct_r3'}, v
    leak.update(variants=VARIANTS, targets_never_inputs=True, shuffle='A permuted within outer-training rows only; test A and all targets untouched')
    (run / 'leakage_audit.json').write_text(json.dumps(leak, indent=1))
    fr.to_csv(run / 'analysis_frame.csv')
    ts = tasks()
    t0 = time.time()
    res = Parallel(n_jobs=64, verbose=0)(delayed(run_task)(fr, f, t) for t in ts for f in sp['folds'])
    pred = pd.DataFrame([r for rs in res for r in rs])
    pred.to_csv(run / 'oof_predictions.csv.gz', index=False)
    meta = dict(splits=str(SPLITS.relative_to(REPO)), splits_sha256=hashlib.sha256(blob.encode()).hexdigest(), n_tasks=len(ts),
                n_fits=len(ts) * len(sp['folds']), wall_seconds=round(time.time() - t0, 1), variants=VARIANTS, n_shuffle=N_SHUFFLE,
                state_schema=state.STATE_SCHEMA_VERSION, python=sys.version.split()[0], sklearn=sklearn.__version__,
                numpy=np.__version__, pandas=pd.__version__, host=platform.node(), cpu_jobs=64,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                state_sha256=hashlib.sha256(Path(state.__file__).read_bytes()).hexdigest())
    (run / 'run_meta.json').write_text(json.dumps(meta, indent=1))
    print(run, meta['n_fits'], 'fits', meta['wall_seconds'], 's')


if __name__ == '__main__':
    main()
