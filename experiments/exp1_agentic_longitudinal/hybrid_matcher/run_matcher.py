#!/usr/bin/env python3
"""Hybrid geometric + appearance lesion matcher with abstention, as pre-specified in the analysis plan (EXPLORATORY).

Pairs = all BL x FU lesions per patient; GT-free pair features; HistGradientBoosting pair classifier; Hungarian assignment with a
no-match threshold tau; patient-grouped 5 x 5 CV on the 274 locked patients (dev patients always in training); tuning (model grid and
tau) on 3 grouped inner folds. Baseline = frozen Exp 2.2 correspondences (state_A). Writes outputs/hybrid_matcher/run/.
  python run_matcher.py [--boot 5000] [--jobs 25] [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / 'outputs/hybrid_matcher'
GT = REPO / 'exp26_clinician_coevolution/data/real/exp22_longitudinal_ct/gt_hidden'
STATE_A = REPO / 'experiments/exp1_agentic_longitudinal/outputs/exp22_baselines/exp22_regen/state_A'
TAUS = [.2, .3, .4, .5, .6, .7]
GRID = [dict(max_iter=200, max_leaf_nodes=15), dict(max_iter=400, max_leaf_nodes=15), dict(max_iter=200, max_leaf_nodes=31),
        dict(max_iter=400, max_leaf_nodes=31)]
GEOM = ['d_reg', 'dx_reg', 'dy_reg', 'dz_reg', 'd_raw', 'd_const', 'rank_reg_b', 'rank_reg_f', 'rank_raw_b', 'rank_raw_f', 'rank_const_b',
        'rank_const_f', 'mutual_reg', 'mutual_const', 'log_vol_ratio', 'log_diam_ratio', 'residual_mm', 'n_bl', 'n_fu', 'n_inliers']
APP = ['d_hu_mean', 'd_hu_sd', 'd_hu_p10', 'd_hu_p90', 'd_elong']


# ------------------------------------------------------------------------------------------------- pairs
def constellation(b, f, tol=10.0):
    """GT-free translation: the BL->FU offset with the most offsets agreeing within tol (raw space); returns (t, n_inliers)."""
    off = (f[None, :, :] - b[:, None, :]).reshape(-1, 3)
    if len(off) == 0:
        return np.zeros(3), 0
    cnt = (np.linalg.norm(off[:, None, :] - off[None, :, :], axis=2) < tol).sum(1)
    k = int(np.argmax(cnt)); inl = np.linalg.norm(off - off[k], axis=1) < tol
    return off[inl].mean(0), int(cnt[k])


def ranks(d, axis):
    return d.argsort(axis=axis).argsort(axis=axis)


def build_pairs(les):
    rows = []
    for pid, g in les.groupby('patient_id'):
        B, F = g[g.side == 'baseline'].reset_index(drop=True), g[g.side == 'followup'].reset_index(drop=True)
        if len(B) == 0 or len(F) == 0:
            continue
        braw, fraw = B[['raw_x', 'raw_y', 'raw_z']].to_numpy(), F[['raw_x', 'raw_y', 'raw_z']].to_numpy()
        breg = B[['reg_x', 'reg_y', 'reg_z']].to_numpy()
        t, ninl = constellation(braw, fraw)
        dreg = np.linalg.norm(breg[:, None] - fraw[None], axis=2)
        draw = np.linalg.norm(braw[:, None] - fraw[None], axis=2)
        dcon = np.linalg.norm(braw[:, None] + t - fraw[None], axis=2)
        rr = {k: (ranks(d, 1), ranks(d, 0)) for k, d in (('reg', dreg), ('raw', draw), ('const', dcon))}
        for i in range(len(B)):
            for j in range(len(F)):
                b, f = B.iloc[i], F.iloc[j]
                rows.append(dict(patient_id=pid, cohort=b.cohort, bl=b.lesion_id, fu=f.lesion_id,
                                 d_reg=dreg[i, j], dx_reg=abs(breg[i, 0] - fraw[j, 0]), dy_reg=abs(breg[i, 1] - fraw[j, 1]),
                                 dz_reg=abs(breg[i, 2] - fraw[j, 2]), d_raw=draw[i, j], d_const=dcon[i, j],
                                 rank_reg_b=rr['reg'][0][i, j], rank_reg_f=rr['reg'][1][i, j], rank_raw_b=rr['raw'][0][i, j],
                                 rank_raw_f=rr['raw'][1][i, j], rank_const_b=rr['const'][0][i, j], rank_const_f=rr['const'][1][i, j],
                                 mutual_reg=int(rr['reg'][0][i, j] == 0 and rr['reg'][1][i, j] == 0),
                                 mutual_const=int(rr['const'][0][i, j] == 0 and rr['const'][1][i, j] == 0),
                                 log_vol_ratio=np.log((f.volume_mm3 + 1) / (b.volume_mm3 + 1)),
                                 log_diam_ratio=np.log((f.diameter_mm + .1) / (b.diameter_mm + .1)), residual_mm=b.residual_mm,
                                 n_bl=len(B), n_fu=len(F), n_inliers=ninl,
                                 d_hu_mean=abs(f.hu_mean - b.hu_mean), d_hu_sd=abs(f.hu_sd - b.hu_sd), d_hu_p10=abs(f.hu_p10 - b.hu_p10),
                                 d_hu_p90=abs(f.hu_p90 - b.hu_p90), d_elong=abs(f.elongation - b.elongation)))
    return pd.DataFrame(rows)


def load():
    les = pd.read_csv(OUT / 'lesions.csv', dtype={'patient_id': str})
    pairs = build_pairs(les)
    gt, cls = {}, {}
    for p in GT.glob('*.json'):
        g = json.loads(p.read_text()); gt[g['patient_id']] = {tuple(x) for x in g['pairs']}
        cls[g['patient_id']] = (g['baseline_class'], g['followup_class'])
    CLS.update(cls)
    ev = pd.read_csv(STATE_A.parent / 'evalx_patients_A.csv', dtype={'patient_id': str})
    GP.update(dict(zip(ev.patient_id, ev.gt_persistent)))
    pairs['y'] = [int((b, f) in gt.get(p, set())) for p, b, f in zip(pairs.patient_id, pairs.bl, pairs.fu)]
    base = {}
    for p in STATE_A.glob('*.json'):
        s = json.loads(p.read_text())
        base[s['patient_id']] = {(f"B{c['baseline_lesion_id']}", f"F{c['followup_lesion_id']}") for c in s['correspondences']
                                 if c.get('followup_lesion_id') is not None and c.get('baseline_lesion_id') is not None}
    return les, pairs, gt, cls, base


# ------------------------------------------------------------------------------------------------- matching
def assign(df, p, tau):
    """Hungarian on 1 - p per patient; keep assigned pairs with p >= tau. Returns {pid: {(bl, fu): p}}."""
    out = {}
    d = df.assign(p=p)
    for pid, g in d.groupby('patient_id'):
        bl, fu = sorted(g.bl.unique()), sorted(g.fu.unique())
        P = np.zeros((len(bl), len(fu)))
        bi, fi = {b: i for i, b in enumerate(bl)}, {f: j for j, f in enumerate(fu)}
        for b, f, pp in zip(g.bl, g.fu, g.p):
            P[bi[b], fi[f]] = pp
        r, c = linear_sum_assignment(1 - P)
        out[pid] = {(bl[i], fu[j]): float(P[i, j]) for i, j in zip(r, c) if P[i, j] >= tau}
    return out


CLS, GP = {}, {}                                                      # filled in load(): candidate classes, GT persistent lesion counts


def counts(pred, gt, pids):
    """Exp 2.2 definition (exp22_full_eval.py): TP = correct pairs; predicted pairs = all predicted pairs; GT persistent = all GT
    persistent lesions (evalx gt_persistent, including partners the segmentation missed)."""
    rows = []
    for p in pids:
        pr = set(pred.get(p, {}))
        rows.append(dict(patient_id=p, tp=len(pr & gt.get(p, set())), pp=len(pr), gp=GP[p]))
    return pd.DataFrame(rows).set_index('patient_id')


def f1(c):
    return 2 * c.tp.sum() / (c.pp.sum() + c.gp.sum())


def fit(X, y, params):
    return HistGradientBoostingClassifier(learning_rate=.05, l2_regularization=1.0, random_state=0, **params).fit(X, y)


def outer_fold(pairs, gt, feats, tr_ids, te_ids):
    tr, te = pairs[pairs.patient_id.isin(tr_ids)], pairs[pairs.patient_id.isin(te_ids)]
    best, bscore = None, -1
    gkf = list(GroupKFold(3).split(tr, tr.y, tr.patient_id))
    for params in GRID:
        oof = np.zeros(len(tr))
        for a, b in gkf:
            oof[b] = fit(tr.iloc[a][feats], tr.iloc[a].y, params).predict_proba(tr.iloc[b][feats])[:, 1]
        for tau in TAUS:
            sc = f1(counts(assign(tr, oof, tau), gt, tr.patient_id.unique()))
            if sc > bscore:
                best, bscore = (params, tau), sc
    m = fit(tr[feats], tr.y, best[0])
    p = m.predict_proba(te[feats])[:, 1]
    return dict(pred=assign(te, p, best[1]), p=pd.Series(p, index=te.index), params=best[0], tau=best[1])


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=5000); ap.add_argument('--jobs', type=int, default=25); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--run-name', default='run', help='output folder name under outputs/hybrid_matcher')
    a = ap.parse_args()
    les, pairs, gt, cls, base = load()
    locked = sorted(les[les.cohort == 'locked'].patient_id.unique()); dev = sorted(les[les.cohort != 'locked'].patient_id.unique())
    run = OUT / a.run_name; run.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(run / 'pairs.csv', index=False)
    t0 = time.time()
    resid = les.drop_duplicates('patient_id').set_index('patient_id').residual_mm
    reg_sens = [p for p in locked if resid[p] > 15]
    cb = counts(base, gt, locked)
    fail = [p for p in locked if cb.at[p, 'gp'] > 0 and cb.at[p, 'tp'] == 0]
    res = dict(n_locked=len(locked), n_pairs=len(pairs), baseline_f1=f1(cb), n_reg_sensitive=len(reg_sens), n_baseline_failures=len(fail))
    print(f"baseline F1 (state_A, locked) {res['baseline_f1']:.4f}  reg-sensitive {len(reg_sens)}  complete failures {len(fail)}", flush=True)
    reps = range(2 if a.quick else 5)
    jobs = []
    for r in reps:
        rng = np.random.default_rng(r); perm = rng.permutation(locked)
        for k in range(5):
            te = sorted(perm[k::5]); tr = sorted(set(locked) - set(te)) + dev
            for name, feats in (('hybrid', GEOM + APP), ('geometric', GEOM)):
                jobs.append((r, k, name, feats, tr, te))
    outs = Parallel(n_jobs=a.jobs)(delayed(outer_fold)(pairs, gt, feats, tr, te) for (r, k, name, feats, tr, te) in jobs)
    preds = {}
    for (r, k, name, *_), o in zip(jobs, outs):
        preds.setdefault((name, r), {}).update(o['pred'])
        preds.setdefault((name, r, 'p'), []).append(o['p'])
    print(f'fitted in {time.time() - t0:.0f}s', flush=True)

    def summary(pred, pids):
        c = counts(pred, gt, pids)
        return dict(f1=f1(c), precision=c.tp.sum() / c.pp.sum(), recall=c.tp.sum() / c.gp.sum(),
                    complete_failure_rate=float(np.mean([(c.at[p, 'gp'] > 0) and c.at[p, 'tp'] == 0 for p in pids if c.at[p, 'gp'] > 0])))

    def class_f1(pred, pids):
        out = {}
        for side, lab, idx in (('bl', 'disappearing', 0), ('fu', 'newly_appearing', 1)):
            tp = fp = fn = 0
            for p in pids:                                            # candidates with a GT lesion and a defined class only
                ids = [l for l, v in cls[p][idx].items() if v in ('persistent', lab)]; matched = {x[0] if side == 'bl' else x[1] for x in pred.get(p, {})}
                for l in ids:
                    pr, tr_ = l not in matched, cls[p][idx][l] == lab
                    tp += pr and tr_; fp += pr and not tr_; fn += (not pr) and tr_
            out[lab] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        return out

    def boot_diff(pa, pb, pids, n, seed=0):
        ca, cb_ = counts(pa, gt, pids), counts(pb, gt, pids)
        rng = np.random.default_rng(seed); arr = np.array(pids); out = []
        for _ in range(n):
            s = rng.choice(arr, len(arr))
            out.append(f1(ca.loc[s]) - f1(cb_.loc[s]))
        return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]

    for name in ('hybrid', 'geometric'):
        per = [summary(preds[(name, r)], locked) for r in reps]
        res[name] = {k: float(np.mean([x[k] for x in per])) for k in per[0]}
        res[name]['per_repeat_f1'] = [x['f1'] for x in per]
        res[name]['delta_f1_vs_baseline'] = res[name]['f1'] - res['baseline_f1']
        res[name]['delta_f1_ci_repeat0'] = boot_diff(preds[(name, 0)], base, locked, 1000 if a.quick else a.boot)
        res[name]['reg_sensitive'] = dict(new=float(np.mean([summary(preds[(name, r)], reg_sens)['f1'] for r in reps])),
                                          baseline=summary(base, reg_sens)['f1'],
                                          ci_repeat0=boot_diff(preds[(name, 0)], base, reg_sens, 1000 if a.quick else a.boot, 1))
        res[name]['baseline_failure_patients'] = dict(new=float(np.mean([summary(preds[(name, r)], fail)['f1'] for r in reps])),
                                                      baseline=summary(base, fail)['f1'])
        res[name]['class_f1'] = {k: float(np.mean([class_f1(preds[(name, r)], locked)[k] for r in reps])) for k in ('disappearing', 'newly_appearing')}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != 'per_repeat_f1'}, default=float), flush=True)
    res['baseline'] = summary(base, locked) | dict(class_f1=class_f1(base, locked), reg_sensitive_f1=summary(base, reg_sens)['f1'])

    # abstention on the hybrid matcher (repeat 0): rank assigned pairs by p
    pr0 = preds[('hybrid', 0)]
    ap_ = pd.DataFrame([dict(patient_id=p, bl=b, fu=f, p=v, correct=int((b, f) in gt.get(p, set()))) for p in locked for (b, f), v in pr0.get(p, {}).items()])
    ab = []
    for budget in (0, .05, .1, .2):
        k = int(round(budget * len(ap_))); s = ap_.sort_values('p')
        review, keep = s.iloc[:k], s.iloc[k:]
        ab.append(dict(review_budget=budget, retained_precision=float(keep.correct.mean()),
                       errors_routed=float((1 - review.correct).sum() / max((1 - ap_.correct).sum(), 1))))
    res['abstention_pairs'] = ab
    flag = {p: any(v < .5 for v in pr0.get(p, {}).values()) or resid[p] > 15 for p in locked}
    c0 = counts(pr0, gt, locked)
    fail_new = [p for p in locked if c0.at[p, 'gp'] > 0 and c0.at[p, 'tp'] == 0]
    res['abstention_patients'] = dict(flagged_fraction=float(np.mean(list(flag.values()))),
                                      failures_flagged=float(np.mean([flag[p] for p in fail_new])) if fail_new else None, n_failures=len(fail_new))
    pall = pd.concat(preds[('hybrid', 0, 'p')]); yv = pairs.loc[pall.index, 'y']
    bins = pd.cut(pall, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    res['calibration'] = dict(brier=float(brier_score_loss(yv, pall)),
                              bins=[dict(bin=str(k), mean_p=float(pall[bins == k].mean()), observed=float(yv[bins == k].mean()), n=int((bins == k).sum()))
                                    for k in bins.cat.categories])
    print('abstention', json.dumps(res['abstention_pairs']), json.dumps(res['abstention_patients']), 'brier', res['calibration']['brier'], flush=True)
    (run / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    rows = [dict(matcher=n, repeat=r, patient_id=p, bl=b, fu=f, p=v) for (n, r) in [k for k in preds if len(k) == 2] for p, d in preds[(n, r)].items()
            for (b, f), v in d.items()]
    pd.DataFrame(rows).to_csv(run / 'predictions.csv', index=False)
    print(f'done {time.time() - t0:.0f}s -> {run}')


if __name__ == '__main__':
    main()
