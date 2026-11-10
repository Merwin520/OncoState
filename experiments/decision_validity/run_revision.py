#!/usr/bin/env python3
"""Part B: certificate re-verification after evidence revision (analysis plan, section 3).
Scenarios: R-real (facts of the real reader R03, 50 clinician-review patients), R-sim (reference-derived facts in random order,
SIMULATED, 274 patients x 5 seeds), R-err (a plausible wrong fact, its retraction, then the reference facts; SIMULATED).
Methods: C0 (marginal cross-fitted q) primary; CA (stratum q) secondary where the stratum pool supports the rank.
    python run_revision.py [--jobs 64] [--smoke]   -> results/revision_events.csv.gz, results/revision_patients.csv"""
import argparse
import csv
import hashlib
import json
import os
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv.common import CFG, REPO, Vault, best, evidence, folds, min_cal, no_reference, prepare, q_tables, roles, stratum  # noqa: E402
from dv.revision import facts_from_pairs, replay, wrong_fact  # noqa: E402

RES = HERE / 'results'
READERS = Path(os.environ.get('READER_DATA', REPO / 'reader_data'))   # clinician reader exports (not distributed)


def r03_pairs():
    parse = lambda s: sorted(tuple(re.match(r'(B\d+)(F\d+)', t).groups()) for t in s.split())  # noqa: E731
    d = json.loads((READERS / 'R03_reviews/reader_pairs_BCDE.json').read_text())
    out = {p: parse(v) for k in d for p, v in d[k].items()}
    for r in csv.DictReader(open(READERS / 'R03_reviews/R03_review_results_A.csv', encoding='utf-8-sig')):
        out.setdefault(r['patient_id'], [])
        if r['row_type'] == 'prior_lesion' and r['clinician_followup_lesion_id']:
            out[r['patient_id']].append((r['baseline_lesion_id'], r['clinician_followup_lesion_id']))
    return {p: sorted(set(v)) for p, v in out.items()}


def job(pid, ev_p, q, scen, seed, method, events, ref_response, vc):
    with no_reference():
        P = prepare(ev_p, CFG['energy']['gate_mm']); _, F0 = best(P)
        rows, summ = replay(P, F0, q, vc, events, ref_response)
    tag = dict(patient_id=pid, scenario=scen, seed=seed, method=method)
    return [dict(tag, **r) for r in rows], dict(tag, n_events=len(events), n_revisions=sum(e[0] == 'add' for e in events), **summ)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=64); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    ev = evidence(); cohort, ext, role = roles(); fold = folds()
    rcfg, vc = CFG['revision'], CFG['verifier']; al = rcfg['alpha']
    if a.smoke:
        cohort, ext = ext, []; fold = {p: i % 5 for i, p in enumerate(cohort)}
    prep = {p: prepare(ev[p], CFG['energy']['gate_mm']) for p in cohort + ext}
    F0 = {p: best(P)[1] for p, P in prep.items()}
    V = Vault(); score = V.scores(prep, F0, cohort + ext)
    st = {p: stratum(ev[p]) for p in cohort + ext}
    QM = q_tables(score, cohort, ext, fold, [al]); QC = q_tables(score, cohort, ext, fold, [al], st)
    qs = {'C0': {p: QM[(al, p)][0] for p in cohort}}
    qs['CA'] = {p: QC[(al, p)][0] for p in cohort if QC[(al, p)][1] >= min_cal(al) and math.isfinite(QC[(al, p)][0])}
    refpairs = {p: V.pairs(p) for p in cohort}
    refresp = {p: V.ref_response(prep[p]) for p in cohort}
    r03 = r03_pairs()
    tasks = []
    for method, qm in qs.items():
        for p in cohort:
            if p not in qm:
                continue
            P = prep[p]
            if p in r03:
                tasks.append((p, 'R-real', 0, method, [('add', x) for x in facts_from_pairs(P, r03[p])]))
            rf = facts_from_pairs(P, refpairs[p])
            for s in rcfg['sim_seeds']:
                o = np.random.default_rng(1000 * s + int(hashlib.sha256(p.encode()).hexdigest()[:8], 16) % 1000).permutation(len(rf))
                order = [rf[i] for i in o]
                tasks.append((p, 'R-sim', s, method, [('add', x) for x in order]))
                if s == rcfg['sim_error_seed']:
                    wf = wrong_fact(P, best(P)[0])
                    if wf is not None and wf not in rf:
                        tasks.append((p, 'R-err', s, method, [('add', wf), ('retract', wf)] + [('add', x) for x in order]))
    print(f'{len(tasks)} replays, setup {time.time() - t0:.0f}s', flush=True)
    out = Parallel(n_jobs=a.jobs, verbose=2, batch_size=4)(delayed(job)(p, ev[p], qs[m][p], sc, s, m, e, refresp[p], vc)
                                                           for p, sc, s, m, e in tasks)
    E = pd.DataFrame([x for rr, _ in out for x in rr]); Pt = pd.DataFrame([s for _, s in out])
    E['role'] = E.patient_id.map(role); Pt['role'] = Pt.patient_id.map(role)
    Pt['ref_response'] = Pt.patient_id.map(refresp); Pt['stratum'] = Pt.patient_id.map(st)
    RES.mkdir(exist_ok=True)
    pre = 'smoke_' if a.smoke else ''
    E.to_csv(RES / f'{pre}revision_events.csv.gz', index=False); Pt.to_csv(RES / f'{pre}revision_patients.csv', index=False)
    meta = dict(n_tasks=len(tasks), runtime_s=time.time() - t0, ca_patients=len(qs['CA']),
                n_r03=len(r03), config_sha256=hashlib.sha256((HERE / 'configs/decision_validity.yaml').read_bytes()).hexdigest())
    (RES / f'{pre}revision_meta.json').write_text(json.dumps(meta, indent=1))
    print('done', len(E), len(Pt), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
