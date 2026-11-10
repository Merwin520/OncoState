#!/usr/bin/env python3
"""4D revision-aware forecasting (experiment_plan.md). Writes results/4D_*.csv.
Same trained predictors for every system (fitted per fold on clean natural states): GBM[state] next-visit FTV and Elastic-Net pCR per
landmark. Systems differ only in how the input state is maintained.  python revision.py [--jobs 48]"""
import argparse
import copy
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import forecast as FC
from check import check
from common import NB, RES, REPO, SEED, T2, F, table_from_memory
from common import ISPY2, M2, _f  # noqa: F401
import sys
sys.path.insert(0, str(REPO / 'experiments/exp28_repeated_synchronization'))
from features.measurement_features import ispy2_cfg  # noqa: E402
from state.synchronization import revise, synchronize  # noqa: E402

warnings.filterwarnings('ignore', category=ConvergenceWarning)
NAMES = ('ftv_pe', 'ftv_ser', 'ld', 'sph', 'bpe')
FACTORS = [1.6, .5, .8, 1.25, 2.0]
REG = {'R0a': ('ftv', 0), 'R0b': ('pcr', 0), 'R1a': ('ftv', 1), 'R1b': ('pcr', 1), 'F2a': ('ftv', 2), 'F2b': ('pcr', 2), 'F3': ('pcr', 3)}
DEPENDS_T1 = {'R0a': False, 'R0b': False, 'R1a': True, 'R1b': True, 'F2a': True, 'F2b': True, 'F3': True}


def pcr_model():
    pipe = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                         LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=.5, max_iter=20000, random_state=0))
    return GridSearchCV(pipe, {'logisticregression__C': [.01, .03, .1, .3, 1]}, cv=3, scoring='roc_auc')


def perturbed(o, f):
    o = copy.deepcopy(o)
    for key in ('ftv_pe_cc', 'ftv_ser_cc'):
        if o['fields'][key]['value'] is not None:
            o['fields'][key]['value'] *= f
    return o


def onco_states(I, p, f, mode):
    """Official OncoState pathway. mode: 'revise' (correction arrives at T2), 'none' (correction lost), 'late' (T1 arrives after T2).
    Returns (versions after each landmark, timing of the correction step, number of revise/synchronize operations at the correction)."""
    cfg = ispy2_cfg(p); s, out, t_corr, ops = None, [], 0., 0
    for k, tp in enumerate(T2):
        o = I.src.observation(p, tp)
        if mode == 'late' and tp == 'T1':
            o = None
        elif tp == 'T1' and o is not None and mode != 'late':
            o = perturbed(o, f)
        if k == 2 and mode == 'revise':
            t0 = time.perf_counter(); corr = I.src.observation(p, 'T1')
            for key in ('ftv_pe_cc', 'ftv_ser_cc'):
                s, _ = revise(s, 'T1', key, corr['fields'][key]['value'], reason='revised record', actor='radiology', source='revised T1 record',
                              cfg=cfg, now='revision'); ops += 1
            t_corr = time.perf_counter() - t0
        s, _ = synchronize(s, tp, o, clinical=I.src.clinical(p) if s is None else None, cfg=cfg, now=f'{mode}-{tp}')
        if k == 2 and mode == 'late':
            t0 = time.perf_counter()
            s, _ = synchronize(s, 'T1', I.src.observation(p, 'T1'), cfg=cfg, event_type='late_arrival', now='late-T1'); ops += 1
            t_corr = time.perf_counter() - t0
        out.append(copy.deepcopy(s))
    return out, t_corr, ops


def src_table(I, p, k, f, corrected, late=False, k_cut=None):
    """Full-history reconstruction: re-read every source record that exists at landmark k (corrected T1 if the correction has arrived)."""
    tab, reads = {}, 1                                          # clinical record
    for tp in T2[:(k_cut if k_cut is not None else k) + 1]:
        o = I.src.observation(p, tp)
        if o is not None and tp == 'T1':
            if late and k < 2:
                o = None
            elif not late and not corrected:
                o = perturbed(o, f)
        if o is not None:
            reads += 1
        tab[tp] = None if o is None else dict(day=o.get('study_day_offset'), **{n: _f(o, key) for key, n, _ in M2})
    return tab, reads


def predict(models, kind, k, tab, clin, arm, dt):
    if kind == 'ftv':
        m, cols = models[('ftv', k)]
        x = pd.DataFrame([FC.feats(tab, T2[k], clin, dt, 'state', arm, [])])[cols]
        return float(m.predict(x.to_numpy(float))[0])
    m, cols = models[('pcr', k)]
    x = pd.DataFrame([F(tab, k, clin)])[cols]
    x['day_latest'] = x['day_latest']
    return float(m.predict_proba(x.to_numpy(float))[0, 1])


def fit_fold(tr_pts, S, C, A, day, y, ypcr):
    models = {}
    for k, (o, t) in enumerate(FC.NEXT):
        X, yv = FC.task_rows(tr_pts, S, C, A, day, y, o, t, 'state', [])
        models[('ftv', k)] = (FC.model('gbm').fit(X.to_numpy(float), yv), list(X.columns))
    for k in range(4):
        X = pd.DataFrame([F(S[p][k], k, C[p]) for p in tr_pts]); yv = np.array([ypcr[p] for p in tr_pts])
        models[('pcr', k)] = (pcr_model().fit(X.to_numpy(float), yv), list(X.columns))
    return models


def run_fold(r, te_pts, pts, S, C, A, day, y, ypcr, mem_rev, mem_late, timing):
    I = FC.ISPY2() if not hasattr(run_fold, 'I') else run_fold.I
    run_fold.I = I
    te = set(te_pts); models = fit_fold([p for p in pts if p not in te], S, C, A, day, y, ypcr)
    rows, costs = [], []
    dt = lambda p, k: (day[p][T2[k + 1]] - day[p][T2[k]]) if k < 3 and day[p][T2[k]] is not None and day[p][T2[k + 1]] is not None else np.nan  # noqa: E731
    for p in te_pts:
        if S[p][1]['T1'] is None:
            continue                                                  # T1 not observed: not eligible
        clin, arm = C[p], A[p]
        def issue(tabs, rid):
            kind, k = REG[rid]
            if kind == 'ftv' and (tabs[k].get(T2[k]) is None or np.isnan(dt(p, k))):
                return np.nan
            return predict(models, kind, k, tabs[k], clin, arm, dt(p, k))
        ref = {rid: issue(S[p], rid) for rid in REG}
        for f in FACTORS:
            vers_r, tc, ops = onco_states(I, p, f, 'revise')
            vers_n, _, _ = onco_states(I, p, f, 'none')
            T_r = [I.table_from_state(vers_r[k], k) for k in range(4)]
            T_n = [I.table_from_state(vers_n[k], k) for k in range(4)]
            pre = {rid: issue(T_r, rid) for rid in ('R0a', 'R0b', 'R1a', 'R1b')}     # issued before the correction
            # OncoState: dependency registry -> recompute only predictions whose input version holds T1, at their original cut-off
            t0 = time.perf_counter()
            T_retro = [I.table_from_state(vers_r[2], k) for k in range(2)]
            onco = dict(pre)
            for rid in ('R1a', 'R1b'):
                onco[rid] = issue(T_retro + [None, None], rid)
            t_onco = time.perf_counter() - t0 + tc
            for rid in ('F2a', 'F2b', 'F3'):
                onco[rid] = issue(T_r, rid)
            # full-history reconstruction: re-read everything after the correction, recompute every registered prediction
            t0 = time.perf_counter(); reads = 0; T_h = []
            for k in range(4):
                tab, rd = src_table(I, p, k, f, corrected=k >= 2); T_h.append(tab); reads += rd if k == 2 else 0
            T_hretro = [src_table(I, p, k, f, corrected=True, k_cut=k)[0] for k in range(2)]
            full = {rid: issue(T_hretro + T_h[2:], rid) if rid in ('R0a', 'R0b', 'R1a', 'R1b') else issue(T_h, rid) for rid in REG}
            t_full = time.perf_counter() - t0
            uncorr = {**pre, **{rid: issue(T_n, rid) for rid in ('F2a', 'F2b', 'F3')}}
            systems = {'reference': ref, 'oncostate': onco, 'full_history': full, 'uncorrected': uncorr}
            tabs_used = {'oncostate': T_r, 'full_history': T_h, 'uncorrected': T_n, 'reference': S[p]}
            if f == 1.6 and p in mem_rev:
                snaps = mem_rev[p]['snapshots']
                T_m = [table_from_memory(snaps[k], k, NAMES) for k in range(4)]
                systems['memory'] = {**{rid: issue(T_m, rid) for rid in REG}}   # no registry: issued predictions are never revisited
                tabs_used['memory'] = T_m
            pv = I.src.observation(p, 'T1')['fields']['ftv_pe_cc']['value']
            for sname, preds in systems.items():
                for rid, v in preds.items():
                    k = REG[rid][1]; tb = tabs_used[sname][k] if k >= 2 else None
                    stale = bool(tb is not None and tb.get('T1') and tb['T1'].get('ftv_pe') is not None and pv and abs(tb['T1']['ftv_pe'] - pv * f) < 1e-6 * max(1, pv)
                                 and f != 1.0)
                    rows.append(dict(scenario='revision', factor=f, repeat=r, patient_id=p, system=sname, rid=rid, pred=v, ref=ref[rid],
                                     pre_correction=pre.get(rid, np.nan), stale_input=float(stale) if k >= 2 else np.nan))
            if r == 0 and f == 1.6:
                costs += [dict(patient_id=p, system='oncostate', records_reread=1, state_ops=ops, model_runs=2, predictions_recomputed=2, sec=t_onco),
                          dict(patient_id=p, system='full_history', records_reread=reads, state_ops=0, model_runs=4, predictions_recomputed=4, sec=t_full)]
                if p in mem_rev:
                    lg = mem_rev[p]['logs'][2]
                    costs.append(dict(patient_id=p, system='memory', records_reread=np.nan, state_ops=1, llm_calls=1 + lg.get('attempt', 0),
                                      llm_tokens=lg.get('tokens'), model_runs=np.nan, predictions_recomputed=0, sec=np.nan))
        # ---------------------------------------------------------------- delayed arrival of T1 (record arrives after T2)
        if S[p][2]['T2'] is not None:
            vers_l, tl, _ = onco_states(I, p, 1.0, 'late')
            T_l = [I.table_from_state(vers_l[k], k) for k in range(4)]
            T_lretro = [I.table_from_state(vers_l[2], k) for k in range(2)]
            onco_l = {rid: issue(T_l, rid) for rid in REG}
            onco_l.update({rid: issue(T_lretro + [None, None], rid) for rid in ('R1a', 'R1b')})
            T_hl = [src_table(I, p, k, 1.0, corrected=True, late=True)[0] for k in range(4)]
            full_l = {rid: issue(T_hl, rid) for rid in REG}
            full_l.update({rid: issue([src_table(I, p, k, 1.0, corrected=True, late=False, k_cut=k)[0] for k in range(2)] + [None, None], rid) for rid in ('R1a', 'R1b')})
            pre_l = {rid: issue(T_l, rid) for rid in ('R0a', 'R0b', 'R1a', 'R1b')}
            sysl = {'reference': ref, 'oncostate': onco_l, 'full_history': full_l, 'uncorrected': {**pre_l, **{rid: issue(T_l, rid) for rid in ('F2a', 'F2b', 'F3')}}}
            if p in mem_late:
                T_ml = [table_from_memory(mem_late[p]['snapshots'][k], k, NAMES) for k in range(4)]
                sysl['memory'] = {rid: issue(T_ml, rid) for rid in REG}
                chron = all((T_ml[k].get('T1') or {}).get('day') is not None for k in (2, 3))
            for sname, preds in sysl.items():
                for rid, v in preds.items():
                    rows.append(dict(scenario='late', factor=np.nan, repeat=r, patient_id=p, system=sname, rid=rid, pred=v, ref=ref[rid],
                                     pre_correction=pre_l.get(rid, np.nan), stale_input=np.nan))
    return rows, costs


def summarize(D, ypcr, y):
    out = []
    for (sc, f, s, rid), g in D.groupby(['scenario', 'factor', 'system', 'rid'], dropna=False):
        g = g.dropna(subset=['pred', 'ref'])
        if g.empty:
            continue
        kind = REG[rid][0]; tol = .01
        diff = np.abs(g.pred - g.ref); bi = FC.boot_idx(len(g))
        rec = dict(scenario=sc, factor=f, system=s, rid=rid, kind=kind, depends_T1=DEPENDS_T1[rid], n=len(g), mean_abs_dev=diff.mean(),
                   recovered=float((diff < tol).mean()), recovered_lo=np.percentile([(diff.to_numpy()[i] < tol).mean() for i in bi], 2.5),
                   recovered_hi=np.percentile([(diff.to_numpy()[i] < tol).mean() for i in bi], 97.5),
                   unchanged_from_issue=float((np.abs(g.pred - g.pre_correction) < 1e-12).mean()) if g.pre_correction.notna().any() else np.nan,
                   stale_input_rate=float(g.stale_input.astype(float).mean()) if g.stale_input.notna().any() else np.nan)
        if kind == 'pcr':
            yy = np.array([ypcr[p] for p in g.patient_id])
            rec.update(auroc=roc_auc_score(yy, g.pred), brier=brier_score_loss(yy, g.pred), auroc_ref=roc_auc_score(yy, g.ref))
        else:
            k = REG[rid][1]; yy = np.array([y[p][T2[k + 1]] for p in g.patient_id]); ok = ~np.isnan(yy)
            rec.update(MAE_log=np.mean(np.abs(g.pred.to_numpy()[ok] - yy[ok])), MAE_log_ref=np.mean(np.abs(g.ref.to_numpy()[ok] - yy[ok])),
                       MAE_cc=np.mean(np.abs(np.expm1(g.pred.to_numpy()[ok]) - np.expm1(yy[ok]))), n_target=int(ok.sum()))
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=25); ap.add_argument('--limit', type=int); a = ap.parse_args()
    check()
    I, folds, pts, S, H, C, A, day, y = FC.load()
    ypcr = {p: int(I.clin.at[p, 'pCR']) for p in pts}
    load_mem = lambda f: {json.loads(l)['patient_id']: json.loads(l) for l in f.read_text().splitlines()} if f.exists() else {}  # noqa: E731
    mem_rev = load_mem(REPO / 'outputs/longitudinal_forecasting/raw/memory_ispy2_revision.jsonl')
    mem_late = load_mem(RES / 'raw/memory_ispy2_late.jsonl')
    splits = [(r, list(g.patient_id.astype(str))) for (r, f), g in folds.groupby(['repeat', 'fold'])]
    if a.limit:
        splits = [(r, te[:a.limit]) for r, te in splits[:2]]
    res = Parallel(n_jobs=a.jobs)(delayed(run_fold)(r, te, pts, S, C, A, day, y, ypcr, mem_rev, mem_late, None) for r, te in splits)
    D = pd.DataFrame([x for rr, _ in res for x in rr]); Cst = pd.DataFrame([x for _, cc in res for x in cc])
    D = D.groupby(['scenario', 'factor', 'system', 'rid', 'patient_id'], dropna=False).mean(numeric_only=True).reset_index()
    D.to_csv(RES / '4D_predictions.csv', index=False); Cst.to_csv(RES / '4D_cost.csv', index=False)
    Sm = summarize(D, ypcr, y); Sm.to_csv(RES / '4D_summary.csv', index=False)
    info = dict(n_memory_revision=len(mem_rev), n_memory_late=len(mem_late), cost=Cst.groupby('system').mean(numeric_only=True).to_dict())
    (RES / '4D_run_info.json').write_text(json.dumps(info, indent=1, default=float))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    print(Sm[(Sm.factor == 1.6) | (Sm.scenario == 'late')].round(3).to_string(index=False))
    print(Cst.groupby('system').mean(numeric_only=True).round(4).to_string())


if __name__ == '__main__':
    main()
