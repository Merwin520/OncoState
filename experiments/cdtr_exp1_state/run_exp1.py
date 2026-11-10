#!/usr/bin/env python3
"""Experiment 1: can structured candidate inference + decision-equivalence verification give a reliable clinical state without
resolving every lesion identity?  274 locked CT patients, no review, no training on the 274.
  python run_exp1.py   -> results/per_patient.json, tables/*.csv (then make_exp1_report.py)"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P1 = REPO / 'experiments/cdtr'
sys.path.insert(0, str(P1)); sys.path.insert(0, str(REPO / 'exp26_clinician_coevolution'))
from cdtr import schema as S  # noqa: E402
from cdtr.conformal import quantile  # noqa: E402
from cdtr.energy import best, min_marginals, prepare  # noqa: E402
from cdtr.sinkhorn import gumbel_samples, sinkhorn_marginals, status_nll  # noqa: E402
from cdtr.verifier import verify  # noqa: E402

CFG = yaml.safe_load((HERE / 'configs/exp1.yaml').read_text())
RES = HERE / 'results'


def batches():
    from adapters.schema_mapping import load_bundle
    from evaluation.clinician_coevolution import SOURCE, make_batches
    from evaluation.cohorts import ALL_REVIEW_PATIENTS
    b = load_bundle(SOURCE)
    return b, make_batches(b['manifest'], json.loads(ALL_REVIEW_PATIENTS.read_text()))


def main():
    t0 = time.time()
    bundle, bt = batches()
    cohort = sorted(p for k in CFG['cohort'] for p in bt[k]); ext = sorted(p for k in CFG['external'] for p in bt[k])
    assert len(cohort) == 274 and len(ext) == 25 and not set(cohort) & set(ext)
    ev = json.loads((P1 / 'results/evidence.json').read_text())
    ref = {p: [tuple(x) for x in v['pairs']] for p, v in json.loads((P1 / 'results/reference_HIDDEN.json').read_text()).items()}
    g = CFG['energy']['gate_mm']
    prep = {p: prepare(ev[p], g) for p in cohort + ext}
    v0 = {p: best(P) for p, P in prep.items()}
    for p, (x, _) in v0.items():
        assert sorted(x) == sorted(prep[p].v0_repository)

    # ---- tau: Gumbel-Sinkhorn temperature from the 25 external patients only
    nll = {}
    for tau in CFG['gumbel_sinkhorn']['tau_grid_mm']:
        tot = [status_nll(prep[p], tau, ref[p]) for p in ext]
        nll[tau] = sum(a for a, _ in tot) / sum(b for _, b in tot)
    tau = min(nll, key=nll.get)
    print('tau NLL', {k: round(v, 4) for k, v in nll.items()}, '-> tau', tau, flush=True)

    # ---- conformal scores and per-patient q (cross-fitted: own fold never used; external-only sensitivity)
    score = {p: prep[p].F(ref[p]) - v0[p][1] for p in cohort + ext}
    rng = np.random.default_rng(CFG['conformal']['fold_seed'])
    perm = rng.permutation(cohort); fold = {p: i % CFG['conformal']['folds'] for i, p in enumerate(perm)}
    alphas = CFG['conformal']['alpha_grid']
    qtab = {}
    for p in cohort:
        cal = [score[o] for o in cohort if fold[o] != fold[p]] + ([score[o] for o in ext] if CFG['conformal']['add_external_to_calibration'] else [])
        for a in alphas:
            qtab[('crossfit', a, p)] = quantile(cal, a)
            qtab[('external_only', a, p)] = quantile([score[o] for o in ext], a)

    # ---- per-patient state inference
    gs_rng = np.random.default_rng(CFG['gumbel_sinkhorn']['seed'])
    rows, per = [], {}
    vc = CFG['verifier']
    for k, p in enumerate(cohort):
        P = prep[p]; x0, F0 = v0[p]
        y0 = P.response(x0)[0]; yref, rinp, _ = P.response(ref[p])
        dm = min_marginals(P, x0)
        Pb, ub, uf = sinkhorn_marginals(P, tau)
        samp = gumbel_samples(P, tau, CFG['gumbel_sinkhorn']['samples'], gs_rng)
        sF = np.array([P.F(s) - F0 for s in samp]); sR = [P.response(s)[0] for s in samp]
        agree = float(np.mean([r == y0 for r in sR]))
        # V0 confidence (patient = least confident follow-up lesion)
        partner = {f: b for b, f in x0}
        bi, fi = {b: i for i, b in enumerate(P.bids)}, {f: j for j, f in enumerate(P.fids)}
        conf = [max(0., 1 - P.d[bi[partner[f]], fi[f]] / g) if f in partner else float(np.clip((P.d[:, fi[f]].min() if P.n else np.inf) / g - 1, 0, 1))
                for f in P.fids]
        # GS lesion-status top-1 probability for calibration
        pf = {b: a for a, b in ref[p]}
        gs_top = []
        for f in P.fids:
            probs = list(Pb[:, fi[f]]) + [uf[fi[f]]]
            j = int(np.argmax(probs)); pred = P.bids[j] if j < P.n else None
            gs_top.append(dict(p=float(probs[j]), correct=int(pred == pf.get(f)), v0_conf=conf[fi[f]], v0_correct=int(partner.get(f) == pf.get(f))))
        rec = dict(patient_id=p, fold=fold[p], n_bl=P.n, n_fu=P.m, v0_response=y0, ref_response=yref, wrong=int(y0 != yref),
                   false_pd=int(y0 == 'PD' and yref != 'PD'), missed_pd=int(y0 != 'PD' and yref == 'PD'), score=score[p],
                   gs_agreement=agree, v0_min_conf=float(min(conf)) if conf else 1.0, residual=ev[p]['registration_residual_mm'])
        for var in ('crossfit', 'external_only'):
            for a in alphas:
                q = qtab[(var, a, p)]
                c = verify(P, q, F0, time_limit=vc['time_limit_s'], tol=vc['energy_tol'], max_cuts=vc['max_nogood_cuts'])
                amb = sum(v <= q + 1e-9 for v in dm.values())
                ins = sF <= q + 1e-9
                samp_reach = sorted({r for r, i in zip(sR, ins) if i})
                rows.append(dict(rec, variant=var, alpha=a, q=q, status=c.status, reachable=';'.join(c.reachable_responses),
                                 n_reachable=len(c.reachable_responses), graph_covered=int(score[p] <= q + 1e-9),
                                 label_covered=int(yref in c.reachable_responses), ambiguous_lesions=int(amb),
                                 identity_certain=int(amb == 0), witness=';'.join(c.witness_lesions),
                                 cex_response=c.counterexample.response if c.counterexample else '', solver_s=c.solver_seconds,
                                 samples_in_set=int(ins.sum()), sample_reachable=';'.join(samp_reach),
                                 sampling_found_cex=int(any(r != y0 for r in samp_reach)),
                                 sound_violation=int(c.status == S.PROVED_EQUIVALENT and any(r != y0 for r in samp_reach))))
        per[p] = dict(gs_top=gs_top)
        if k % 25 == 0:
            print(k, p, round(time.time() - t0, 1), flush=True)
    df = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / 'per_patient_states.csv', index=False)
    (RES / 'gs_lesion_calibration.json').write_text(json.dumps(per))
    meta = dict(tau=tau, tau_nll=nll, n_cohort=len(cohort), n_external=len(ext), runtime_s=time.time() - t0,
                cohort_sha256=hashlib.sha256(','.join(cohort).encode()).hexdigest(),
                config_sha256=hashlib.sha256((HERE / 'configs/exp1.yaml').read_bytes()).hexdigest(),
                external_q={str(a): quantile([score[o] for o in ext], a) for a in alphas},
                crossfit_q_by_fold={str(a): sorted({round(qtab[('crossfit', a, p)], 3) for p in cohort}) for a in alphas})
    (RES / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != 'tau_nll'}, indent=1))


if __name__ == '__main__':
    main()
