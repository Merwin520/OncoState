#!/usr/bin/env python3
"""Part A: coverage-aware decision acceptance vs original CDT-R and controls (analysis plan, section 2).
Step 1 (reference allowed, Vault only): nonconformity scores -> per-patient q tables (own fold excluded).
Step 2 (reference blocked): verifier decisions for every patient x method x alpha.
Step 3 (Vault): rule-derived reference responses and set coverage for scoring.
    python run_acceptance.py [--jobs 64] [--smoke]   -> results/acceptance_per_patient.csv, results/acceptance_meta.json"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv.common import CFG, REPO, Vault, best, evidence, folds, no_reference, prepare, q_tables, roles, stratum  # noqa: E402
from dv.accept import decide  # noqa: E402

RES = HERE / 'results'
HYB = REPO / 'experiments/exp1_hybrid_cdtr'


def job(pid, ev_p, specs, vc, hyb=None):
    """specs: list of (method, alpha, q, n_pool, enforce_pool, energy). Runs with reference access blocked."""
    with no_reference():
        P = prepare(ev_p, CFG['energy']['gate_mm']); _, F0 = best(P)
        PH = F0H = None
        if hyb is not None:
            sys.path.insert(0, str(HYB))
            from run_hybrid_cdtr import prep as hprep
            PH, _ = hprep(pid, ev_p, hyb); _, F0H = best(PH)
        out = []
        for method, a, q, n_pool, enforce, energy in specs:
            PP, FF = (PH, F0H) if energy == 'hybrid' else (P, F0)
            t = time.perf_counter()
            d = decide(PP, FF, q, n_pool, a, vc, enforce)
            out.append(dict(patient_id=pid, method=method, alpha=a, q=q, n_pool=n_pool, energy=energy, wall_s=time.perf_counter() - t, **d))
        return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=64); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--hybrid-repeats', default='0,1,2,3,4')
    a = ap.parse_args()
    t0 = time.time()
    ev = evidence(); cohort, ext, role = roles(); fold = folds()
    acfg, vc = CFG['acceptance'], CFG['verifier']
    if a.smoke:                       # pipeline check on the 25 development patients only: they play the cohort, folds by index
        cohort, ext = ext, []
        fold = {p: i % 5 for i, p in enumerate(cohort)}
    prep = {p: prepare(ev[p], CFG['energy']['gate_mm']) for p in cohort + ext}
    F0 = {p: best(P)[1] for p, P in prep.items()}
    V = Vault()
    score = V.scores(prep, F0, cohort + ext)

    # strata (reference-free)
    st_primary = {p: stratum(ev[p]) for p in cohort + ext}
    st_cut = {c: {p: stratum(ev[p], cut=c) for p in cohort + ext} for c in acfg['strata']['cut_sensitivity_mm']}
    alphas_marg = sorted(set(acfg['expanded_alpha_grid']) | {acfg['alpha_primary']} | set(acfg['alpha_sensitivity']) | {.02})
    alphas_ca = sorted({acfg['alpha_primary']} | set(acfg['alpha_sensitivity']) | set(acfg['expanded_alpha_grid']))
    QM = q_tables(score, cohort, ext, fold, alphas_marg)
    QC = q_tables(score, cohort, ext, fold, alphas_ca, st_primary)
    QS = {c: q_tables(score, cohort, ext, fold, [acfg['alpha_primary']], s) for c, s in st_cut.items()}
    # tertile strata: edges from each patient's own calibration pool residuals (label-free)
    QT, st_tert = {}, {}
    for p in cohort:
        pool = [o for o in cohort if fold[o] != fold[p]] + list(ext)
        res = [ev[o]['registration_residual_mm'] for o in pool if stratum(ev[o]) != 'single']
        edges = np.quantile(res, [1 / 3, 2 / 3])
        sp = stratum(ev[p], edges=edges)
        st_tert[p] = sp
        pool_s = pool if sp == 'single' else [o for o in pool if stratum(ev[o], edges=edges) == sp]
        from dv.common import quantile
        QT[p] = (quantile([score[o] for o in pool_s], acfg['alpha_primary']), len(pool_s))

    # hybrid energy (C6 reproduction, C7 hybrid + CA): pools = other-fold cohort only (development in-sample for the hybrid)
    hyb_specs = {}
    reps = [] if a.smoke else [int(x) for x in a.hybrid_repeats.split(',')]
    oof = pd.read_csv(HYB / 'results/hybrid_oof.csv.gz', dtype={'patient_id': str}) if reps else None
    hyb_data = {}
    if reps:
        sys.path.insert(0, str(HYB))
        from run_hybrid_cdtr import prep as hprep
        from dv.common import quantile
        for r in reps:
            o = oof[oof.repeat == r]; og = dict(tuple(o.groupby('patient_id'))); empty = o.iloc[:0]
            hd = {p: og.get(p, empty) for p in cohort}
            hyb_data[r] = hd
            hs = {}
            for p in cohort:
                PH, _ = hprep(p, ev[p], hd[p]); _, f0 = best(PH); hs[p] = PH.F(V.pairs(p)) - f0
            for p in cohort:
                pool = [x for x in cohort if fold[x] != fold[p]]
                pool_ca = pool if st_primary[p] == 'single' else [x for x in pool if st_primary[x] == st_primary[p]]
                hyb_specs.setdefault(p, []).append((f'C6_hybrid_r{r}', acfg['alpha_primary'], quantile([hs[x] for x in pool], acfg['alpha_primary']), len(pool), False, 'hybrid', r))
                hyb_specs[p].append((f'C7_hybrid_CA_r{r}', acfg['alpha_primary'], quantile([hs[x] for x in pool_ca], acfg['alpha_primary']), len(pool_ca), True, 'hybrid', r))
            hyb_data[r]['_scores'] = hs

    specs = {}
    for p in cohort:
        s = [('C0', al, *QM[(al, p)], False, 'v0') for al in alphas_marg]
        s += [('CA', al, *QC[(al, p)], True, 'v0') for al in alphas_ca]
        s += [(f'CA_cut{int(c)}', acfg['alpha_primary'], *QS[c][(acfg['alpha_primary'], p)], True, 'v0') for c in st_cut]
        s += [('CA_tertile', acfg['alpha_primary'], *QT[p], True, 'v0')]
        specs[p] = s
    print(f'step 1 done {time.time() - t0:.0f}s; {sum(map(len, specs.values()))} v0 specs', flush=True)

    res = Parallel(n_jobs=a.jobs, verbose=2)(delayed(job)(p, ev[p], specs[p], vc) for p in cohort)
    rows = [x for rr in res for x in rr]
    for r in reps:
        hres = Parallel(n_jobs=a.jobs, verbose=1)(delayed(job)(p, ev[p], [s[:6] for s in hyb_specs[p] if s[6] == r], vc, hyb_data[r][p])
                                                  for p in cohort)
        rows += [x for rr in hres for x in rr]
        print(f'hybrid repeat {r} done {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)

    # step 3: scoring (Vault)
    refresp = {p: V.ref_response(prep[p]) for p in cohort}
    df['ref_response'] = df.patient_id.map(refresp)
    df['wrong'] = (df.response != df.ref_response).astype(int)
    df['false_pd'] = ((df.response == 'PD') & (df.ref_response != 'PD')).astype(int)
    df['missed_pd'] = ((df.response != 'PD') & (df.ref_response == 'PD')).astype(int)
    hsc = {r: hyb_data[r]['_scores'] for r in reps}
    sc_row = [hsc[int(m.split('_r')[-1])][p] if e == 'hybrid' else score[p] for p, m, e in zip(df.patient_id, df.method, df.energy)]
    df['ref_score'] = sc_row
    df['ref_included'] = (df.ref_score <= df.q + 1e-9).astype(int)
    df['stratum'] = df.patient_id.map(st_primary)
    df['stratum_tertile'] = df.patient_id.map(st_tert)
    for c, s in st_cut.items():
        df[f'stratum_cut{int(c)}'] = df.patient_id.map(s)
    df['residual_mm'] = df.patient_id.map(lambda p: ev[p]['registration_residual_mm'])
    df['role'] = df.patient_id.map(role)
    RES.mkdir(exist_ok=True)
    out = RES / ('smoke_acceptance.csv' if a.smoke else 'acceptance_per_patient.csv')
    df.to_csv(out, index=False)
    meta = dict(n_cohort=len(cohort), n_external=len(ext), runtime_s=time.time() - t0, jobs=a.jobs,
                hybrid_repeats=reps, cohort_sha256=hashlib.sha256(','.join(cohort).encode()).hexdigest(),
                config_sha256=hashlib.sha256((HERE / 'configs/decision_validity.yaml').read_bytes()).hexdigest(),
                stratum_counts=pd.Series(st_primary).loc[cohort].value_counts().to_dict(),
                python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=__import__('scipy').__version__)
    (RES / ('smoke_acceptance_meta.json' if a.smoke else 'acceptance_meta.json')).write_text(json.dumps(meta, indent=1))
    print('wrote', out, len(df), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
