#!/usr/bin/env python3
"""Alpha trade-off and stratification ablation (post hoc analysis-plan amendment): C0 / CA / ablation strata over the alpha grid (V0 energy), C6 / C7 over the grid
for hybrid repeats 0-4, and 100 random-strata draws at alpha 0.10. Same verifier, rule engine, folds and reference isolation as
run_acceptance.py.
    python run_alpha_tradeoff.py [--jobs 64]   -> results/alpha_tradeoff_per_patient.csv.gz, results/alpha_tradeoff_meta.json"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv.common import CFG, Vault, best, evidence, folds, prepare, q_tables, quantile, roles, stratum  # noqa: E402
from run_acceptance import HYB, job  # noqa: E402

G = [.005, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .12, .15, .20, .25, .30]
N_RAND = 100


def v0_min_conf(P, g=30.0):
    """V0 lesion-matching confidence (as in cdtr_exp1_state/run_exp1.py): least confident follow-up lesion of the V0 matching."""
    x0, _ = best(P)
    partner = {f: b for b, f in x0}
    bi, fi = {b: i for i, b in enumerate(P.bids)}, {f: j for j, f in enumerate(P.fids)}
    conf = [max(0., 1 - P.d[bi[partner[f]], fi[f]] / g) if f in partner else float(np.clip((P.d[:, fi[f]].min() if P.n else np.inf) / g - 1, 0, 1))
            for f in P.fids]
    return float(min(conf)) if conf else 1.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=64); a = ap.parse_args()
    t0 = time.time()
    ev = evidence(); cohort, ext, role = roles(); fold = folds()
    vc = CFG['verifier']; g = CFG['energy']['gate_mm']
    allp = cohort + ext
    prep = {p: prepare(ev[p], g) for p in allp}
    F0 = {p: best(P)[1] for p, P in prep.items()}
    V = Vault(); score = V.scores(prep, F0, allp)

    # ---------------------------------------------------------------- strata (label-free)
    st = {p: stratum(ev[p]) for p in allp}
    ns = [p for p in allp if st[p] != 'single']
    share = sum(st[p] == 'degraded' for p in ns) / len(ns)
    n_deg = int(round(share * len(ns)))
    conf = {p: v0_min_conf(prep[p], g) for p in allp}
    nles = {p: prep[p].n + prep[p].m for p in allp}

    def by_rank(val, high_is_degraded):
        o = sorted(ns, key=lambda p: (-val[p] if high_is_degraded else val[p], p))
        deg = set(o[:n_deg])
        return {p: ('single' if st[p] == 'single' else ('degraded' if p in deg else 'reliable')) for p in allp}
    strata = {'CA': st, 'CA_conf': by_rank(conf, False), 'CA_nles': by_rank(nles, True)}
    rng = np.random.default_rng(CFG['bootstrap']['seed'])
    rand = []
    for r in range(N_RAND):
        deg = set(rng.choice(ns, n_deg, replace=False))
        rand.append({p: ('single' if st[p] == 'single' else ('degraded' if p in deg else 'reliable')) for p in allp})

    specs = {p: [] for p in cohort}
    QM = q_tables(score, cohort, ext, fold, G)
    for p in cohort:
        specs[p] += [('C0', al, *QM[(al, p)], False, 'v0') for al in G]
    for name, s in strata.items():
        Q = q_tables(score, cohort, ext, fold, G, s)
        for p in cohort:
            specs[p] += [(name, al, *Q[(al, p)], True, 'v0') for al in G]
    for r, s in enumerate(rand):
        Q = q_tables(score, cohort, ext, fold, [.10], s)
        for p in cohort:
            specs[p].append((f'CA_rand{r:03d}', .10, *Q[(.10, p)], True, 'v0'))
    print(f'V0 specs {sum(map(len, specs.values()))}, setup {time.time() - t0:.0f}s', flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=1)(delayed(job)(p, ev[p], specs[p], vc) for p in cohort)
    rows = [x for rr in res for x in rr]
    strat_of = {'C0': st, **strata, **{f'CA_rand{r:03d}': s for r, s in enumerate(rand)}}

    # ---------------------------------------------------------------- hybrid C6 / C7 over the grid
    sys.path.insert(0, str(HYB))
    from run_hybrid_cdtr import prep as hprep
    oof = pd.read_csv(HYB / 'results/hybrid_oof.csv.gz', dtype={'patient_id': str})
    hscore = {}
    for r in range(5):
        o = oof[oof.repeat == r]; og = dict(tuple(o.groupby('patient_id'))); empty = o.iloc[:0]
        hd = {p: og.get(p, empty) for p in cohort}
        hs = {}
        for p in cohort:
            PH, _ = hprep(p, ev[p], hd[p]); _, f0 = best(PH); hs[p] = PH.F(V.pairs(p)) - f0
        hscore[r] = hs
        hspec = {}
        for p in cohort:
            pool = [x for x in cohort if fold[x] != fold[p]]
            pool_ca = pool if st[p] == 'single' else [x for x in pool if st[x] == st[p]]
            hspec[p] = [(f'C6_r{r}', al, quantile([hs[x] for x in pool], al), len(pool), False, 'hybrid') for al in G]
            hspec[p] += [(f'C7_r{r}', al, quantile([hs[x] for x in pool_ca], al), len(pool_ca), True, 'hybrid') for al in G]
        hres = Parallel(n_jobs=a.jobs, verbose=0)(delayed(job)(p, ev[p], hspec[p], vc, hd[p]) for p in cohort)
        rows += [x for rr in hres for x in rr]
        print(f'hybrid repeat {r} done {time.time() - t0:.0f}s', flush=True)
        strat_of[f'C6_r{r}'] = st; strat_of[f'C7_r{r}'] = st

    # ---------------------------------------------------------------- scoring (Vault)
    D = pd.DataFrame(rows)
    refresp = {p: V.ref_response(prep[p]) for p in cohort}
    D['ref_response'] = D.patient_id.map(refresp)
    D['wrong'] = (D.response != D.ref_response).astype(int)
    D['false_pd'] = ((D.response == 'PD') & (D.ref_response != 'PD')).astype(int)
    sc = [hscore[int(m.split('_r')[-1])][p] if e == 'hybrid' else score[p] for p, m, e in zip(D.patient_id, D.method, D.energy)]
    D['ref_included'] = (np.array(sc) <= D.q.to_numpy() + 1e-9).astype(int)
    D['stratum_used'] = [strat_of[m][p] for m, p in zip(D.method, D.patient_id)]
    D['stratum'] = D.patient_id.map(st)
    D['residual_mm'] = D.patient_id.map(lambda p: ev[p]['registration_residual_mm'])
    D = D.drop(columns=['reachable'])
    D.to_csv(HERE / 'results/alpha_tradeoff_per_patient.csv.gz', index=False)
    # label-free stratum variables + calibration-only scores for the coverage-gap diagnostic
    pd.DataFrame(dict(patient_id=allp, stratum=[st[p] for p in allp], conf=[conf[p] for p in allp], nles=[nles[p] for p in allp],
                      in_cohort=[p in set(cohort) for p in allp], fold=[fold.get(p, -1) for p in allp],
                      v0_score=[score[p] for p in allp], **{f'hyb_score_r{r}': [hscore[r].get(p, np.nan) for p in allp] for r in range(5)})
                 ).to_csv(HERE / 'results/alpha_tradeoff_patient_vars.csv', index=False)
    meta = dict(grid=G, n_random=N_RAND, degraded_share=share, n_degraded_nonsingle=n_deg,
                runtime_s=time.time() - t0, rows=len(D))
    (HERE / 'results/alpha_tradeoff_meta.json').write_text(json.dumps(meta, indent=1))
    print('wrote', len(D), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
