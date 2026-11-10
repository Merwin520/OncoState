#!/usr/bin/env python3
"""Reproduce the frozen hybrid-matcher run (outputs/hybrid_matcher/run) and export out-of-fold pair probabilities for
ALL candidate pairs, plus the outer-fold no-match threshold tau, for every repeat. Checks that the hybrid assignment rule applied to the
exported probabilities reproduces the saved predictions.csv in every patient x repeat.
    python export_hybrid_energy.py [--jobs 20]   -> results/hybrid_oof.parquet (patient_id, bl, fu, repeat, p, tau)"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HM = REPO / 'experiments/exp1_agentic_longitudinal/hybrid_matcher'
RUN = REPO / 'outputs/hybrid_matcher/run'
sys.path.insert(0, str(HM))
import run_matcher as M  # noqa: E402


def fold_job(pairs, gt, tr, te, gp, cls):
    M.GP.update(gp); M.CLS.update(cls)                       # module globals filled by load() in the parent; workers re-import the module
    o = M.outer_fold(pairs, gt, M.GEOM + M.APP, tr, te)
    return o['p'], o['tau'], o['pred']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=20); a = ap.parse_args()
    les, pairs, gt, cls, base = M.load()
    locked = sorted(les[les.cohort == 'locked'].patient_id.unique()); dev = sorted(les[les.cohort != 'locked'].patient_id.unique())
    jobs = []
    for r in range(5):                                       # identical fold construction to run_matcher.main
        perm = np.random.default_rng(r).permutation(locked)
        for k in range(5):
            te = sorted(perm[k::5]); tr = sorted(set(locked) - set(te)) + dev
            jobs.append((r, k, tr, te))
    outs = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fold_job)(pairs, gt, tr, te, dict(M.GP), dict(M.CLS)) for (r, k, tr, te) in jobs)
    rows, preds = [], {}
    for (r, k, tr, te), (p, tau, pred) in zip(jobs, outs):
        d = pairs.loc[p.index, ['patient_id', 'bl', 'fu']].assign(repeat=r, fold=k, p=p.values, tau=tau)
        rows.append(d)
        for pid, dd in pred.items():
            preds[(r, pid)] = dd
    oof = pd.concat(rows, ignore_index=True)
    saved = pd.read_csv(RUN / 'predictions.csv', dtype={'patient_id': str})
    saved = saved[saved.matcher == 'hybrid']
    ok = tot = 0; maxdiff = 0.0
    for (r, pid), dd in preds.items():
        s = saved[(saved.repeat == r) & (saved.patient_id == pid)]
        a_ = {(b, f): v for b, f, v in zip(s.bl, s.fu, s.p)}
        tot += 1; ok += set(a_) == set(dd)
        for kk in set(a_) & set(dd):
            maxdiff = max(maxdiff, abs(a_[kk] - dd[kk]))
    print(f'reproduced assignments: {ok}/{tot}; max |p diff| on shared pairs {maxdiff:.2e}', flush=True)
    assert ok == tot and maxdiff < 1e-9, 'hybrid run not reproduced exactly'
    HERE.joinpath('results').mkdir(exist_ok=True)
    oof.to_csv(HERE / 'results/hybrid_oof.csv.gz', index=False)
    (HERE / 'results/export_meta.json').write_text(json.dumps(dict(
        reproduced=f'{ok}/{tot}', max_p_diff=maxdiff, n_rows=len(oof), source_run=str(RUN.relative_to(REPO)),
        predictions_sha256=hashlib.sha256((RUN / 'predictions.csv').read_bytes()).hexdigest(),
        tau_by_repeat_fold={f'{r}/{k}': float(t) for (r, k, *_), (_, t, _) in zip(jobs, outs)}), indent=1))
    print('wrote', HERE / 'results/hybrid_oof.csv.gz', len(oof), 'rows')


if __name__ == '__main__':
    main()
