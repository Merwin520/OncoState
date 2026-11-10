#!/usr/bin/env python3
"""Hybrid matcher + CDT-R on the 274 locked Exp 1 patients (pre-specified analysis plan).
Energy: cost_ij = -(logit p_ij - logit tau_fold) from the exported out-of-fold hybrid probabilities; unmatched = 0.
Calibration: cross-fitted split conformal with the Exp 1 patient folds; development patients excluded (in-sample for the hybrid).
    python run_hybrid_cdtr.py [--jobs 24]   -> results/per_patient_hybrid.csv, results/meta.json"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P1 = REPO / 'experiments/cdtr'
sys.path.insert(0, str(P1))
from cdtr import schema as S  # noqa: E402
from cdtr.conformal import quantile  # noqa: E402
from cdtr.energy import Prepared, best, min_marginals  # noqa: E402
from cdtr.rules import tenths  # noqa: E402
from cdtr.verifier import verify  # noqa: E402

ALPHAS = [.02, .05, .10, .15, .20, .30]
EPS = 1e-6
logit = lambda x: np.log(x / (1 - x))  # noqa: E731


def prep(pid, ev, oof_p):
    bl, fu = ev['baseline'], ev['followup']
    bids, fids = [l['lesion_id'] for l in bl], [l['lesion_id'] for l in fu]
    cost = np.zeros((len(bids), len(fids)))
    pm = np.full((len(bids), len(fids)), np.nan)
    if len(oof_p):
        tau = float(oof_p.tau.iloc[0])
        bi, fi = {b: i for i, b in enumerate(bids)}, {f: j for j, f in enumerate(fids)}
        for b, f, p in zip(oof_p.bl, oof_p.fu, oof_p.p):
            pc = float(np.clip(p, EPS, 1 - EPS))
            cost[bi[b], fi[f]] = -(logit(pc) - logit(tau)); pm[bi[b], fi[f]] = p
        assert not np.isnan(pm).any(), pid
    P = Prepared(pid, bids, fids, cost, 0.0, {l['lesion_id']: tenths(l['diameter_mm']) for l in bl},
                 {l['lesion_id']: tenths(l['diameter_mm']) for l in fu})
    return P, pm


def patient_job(pid, ev, oof_p, ref, hyb_pred, qs):
    P, pm = prep(pid, ev, oof_p)
    x0, F0 = best(P)
    y0 = P.response(x0)[0]; yref = P.response(ref)[0]
    bi, fi = {b: i for i, b in enumerate(P.bids)}, {f: j for j, f in enumerate(P.fids)}
    conf = min([pm[bi[b], fi[f]] for b, f in x0], default=1.0)
    dm = min_marginals(P, x0) if P.n and P.m else {}
    score = P.F(ref) - F0
    rows = []
    for a, q in qs.items():
        t0 = time.time()
        c = verify(P, q, F0, time_limit=30.0, tol=1e-6, max_cuts=5)
        rows.append(dict(patient_id=pid, alpha=a, q=q, status=c.status, reachable=';'.join(c.reachable_responses), n_reachable=len(c.reachable_responses),
                         ref_included=int(score <= q + 1e-9), label_covered=int(yref in c.reachable_responses),
                         ambiguous_lesions=int(sum(v <= q + 1e-9 for v in dm.values())), solver_s=time.time() - t0))
    base = dict(patient_id=pid, hyb_response=y0, ref_response=yref, wrong=int(y0 != yref), false_pd=int(y0 == 'PD' and yref != 'PD'),
                score=score, hyb_min_conf=float(conf), argmin_eq_hybrid=int(sorted(x0) == sorted(hyb_pred)), n_bl=P.n, n_fu=P.m,
                hyb_pairs=';'.join(f'{b}-{f}' for b, f in sorted(x0)))
    return [dict(base, **r) for r in rows]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=24); ap.add_argument('--repeats', default='0,1,2,3,4')
    a = ap.parse_args()
    t_start = time.time()
    st = pd.read_csv(REPO / 'experiments/cdtr_exp1_state/results/per_patient_states.csv', dtype={'patient_id': str})
    folds = st.drop_duplicates('patient_id').set_index('patient_id').fold
    cohort = sorted(folds.index)
    ev = json.loads((P1 / 'results/evidence.json').read_text())
    ref = {p: [tuple(x) for x in v['pairs']] for p, v in json.loads((P1 / 'results/reference_HIDDEN.json').read_text()).items()}
    oof = pd.read_csv(HERE / 'results/hybrid_oof.csv.gz', dtype={'patient_id': str})
    pred = pd.read_csv(REPO / 'outputs/hybrid_matcher/run/predictions.csv', dtype={'patient_id': str})
    pred = pred[pred.matcher == 'hybrid']
    out = []
    for r in [int(x) for x in a.repeats.split(',')]:
        o = oof[oof.repeat == r]; og = dict(tuple(o.groupby('patient_id')))
        pr = pred[pred.repeat == r]; pg = {p: list(zip(g.bl, g.fu)) for p, g in pr.groupby('patient_id')}
        empty = o.iloc[:0]
        # nonconformity scores (energy only; no verifier) for cross-fitted calibration
        sc = {}
        for p in cohort:
            P, _ = prep(p, ev[p], og.get(p, empty)); _, F0 = best(P); sc[p] = P.F(ref[p]) - F0
        qtab = {p: {al: quantile([sc[o_] for o_ in cohort if folds[o_] != folds[p]], al) for al in ALPHAS} for p in cohort}
        res = Parallel(n_jobs=a.jobs, verbose=2)(delayed(patient_job)(p, ev[p], og.get(p, empty), ref[p], pg.get(p, []), qtab[p]) for p in cohort)
        df = pd.DataFrame([x for rr in res for x in rr]).assign(repeat=r)
        out.append(df)
        prim = df[np.isclose(df.alpha, .10)]
        c_ = prim.status.eq(S.PROVED_EQUIVALENT)
        print(f'repeat {r}: certified {c_.sum()}/274, wrong among certified {prim[c_].wrong.sum()}, coverage {prim.ref_included.mean():.3f}, '
              f'argmin==hybrid {prim.argmin_eq_hybrid.mean():.3f}, {time.time() - t_start:.0f}s', flush=True)
        pd.concat(out).to_csv(HERE / 'results/per_patient_hybrid.csv', index=False)
    meta = dict(runtime_s=time.time() - t_start, alphas=ALPHAS, n_cohort=len(cohort),
                calibration='crossfit, Exp 1 folds, development patients excluded', energy='-(logit p - logit tau_fold), unmatched 0')
    (HERE / 'results/meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
