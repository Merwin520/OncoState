#!/usr/bin/env python3
"""Exp 3.0: downstream clinical impact of clinician corrections, as pre-specified in the analysis plan.
Reuses the unchanged Exp 2.6 package (batches, candidate repair, validation gate, one-page decision resolution).
  python run_impact.py --events <directory of reviewer folders>"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PKG = REPO / 'exp26_clinician_coevolution'
OUT = REPO / 'outputs/exp30'
os.chdir(PKG); sys.path.insert(0, str(PKG))
from adapters.schema_mapping import load_bundle  # noqa: E402
from app import one_page as op  # noqa: E402
from configs.loader import load_config  # noqa: E402
from corrections.correction_event import is_correction  # noqa: E402
from evaluation.clinician_coevolution import SOURCE, consensus_events, make_batches, reviewer_events  # noqa: E402
from evaluation.cohorts import ALL_REVIEW_PATIENTS, cohort  # noqa: E402
from evaluation.gt_access import load_gt  # noqa: E402
from repair.build_candidate import fit_candidate  # noqa: E402
from repair.matching_repair import CentroidHungarianMatcher  # noqa: E402
from repair.validation_gate import run_gate  # noqa: E402


def assess(case, pairs):
    """RECIST 1.1-style response from lesion pairs; diameters fixed by the case."""
    bl = {l['lesion_id']: l['diameter_mm'] for l in case['baseline_lesions']}
    fu = {l['lesion_id']: l['diameter_mm'] for l in case['followup_lesions']}
    partner = {a: b for a, b in pairs}
    targets = [a for a, d in sorted(bl.items(), key=lambda kv: -kv[1]) if d >= 10][:5]
    s0 = sum(bl[a] for a in targets)
    s1 = sum(fu[partner[a]] if partner.get(a) in fu else 0. for a in targets)
    matched_fu = {b for _, b in pairs}
    new = any(d >= 10 for b, d in fu.items() if b not in matched_fu)
    if not targets:
        resp = 'PD' if new else 'NE'
    elif new or (s1 - s0 >= .2 * s0 and s1 - s0 >= 5):
        resp = 'PD'
    elif s1 == 0:
        resp = 'CR'
    elif s1 <= .7 * s0:
        resp = 'PR'
    else:
        resp = 'SD'
    tb, tf = sum(bl.values()), sum(fu.values())
    return dict(response=resp, new_lesion=new, sum_bl=s0, sum_fu=s1)


def outcome_rows(cases, gt, pids, preds, condition, source):
    rows = []
    for p in pids:
        ref = assess(cases[p], list(gt[p]['pairs'])); pr = assess(cases[p], list(preds[p]))
        rows.append(dict(patient_id=p, condition=condition, event_source=source, ref_response=ref['response'], pred_response=pr['response'],
                         response_wrong=int(pr['response'] != ref['response']),
                         pd_wrong=int((pr['response'] == 'PD') != (ref['response'] == 'PD')),
                         false_pd=int(pr['response'] == 'PD' and ref['response'] != 'PD'),
                         missed_pd=int(pr['response'] != 'PD' and ref['response'] == 'PD'),
                         missed_new=int(ref['new_lesion'] and not pr['new_lesion']), false_new=int(pr['new_lesion'] and not ref['new_lesion']),
                         ref_new=int(ref['new_lesion'])))
    return rows


def paired(df, cond, base, B=2000, seed=20261011):
    a = df[df.condition == base].set_index('patient_id'); b = df[df.condition == cond].set_index('patient_id').loc[a.index]
    out = {}
    for ep in ('response_wrong', 'pd_wrong', 'missed_new', 'false_new'):
        x, z = a[ep].to_numpy(), b[ep].to_numpy()
        fixed, intro = int(((x == 1) & (z == 0)).sum()), int(((x == 0) & (z == 1)).sum())
        rng = np.random.default_rng(seed); n = len(x)
        bs = [(z[i].mean() - x[i].mean()) for i in (rng.integers(0, n, n) for _ in range(B))]
        out[ep] = dict(base_rate=x.mean(), cond_rate=z.mean(), diff=z.mean() - x.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                       fixed=fixed, introduced=intro, mcnemar_p=binomtest(fixed, fixed + intro, .5).pvalue if fixed + intro else 1.0)
    changed = int((a.pred_response != b.pred_response).sum())
    return changed, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--events', type=Path, required=True)
    a = ap.parse_args()
    run = OUT / 'run'; run.mkdir(parents=True, exist_ok=True)
    cfg = load_config('configs/experiment_real.yaml')
    seed = cfg['random_seed']
    bundle = load_bundle(SOURCE); cases, gt = bundle['cases'], load_gt(SOURCE)
    batches = make_batches(bundle['manifest'], json.loads(ALL_REVIEW_PATIENTS.read_text(encoding='utf-8')))
    co = cohort('50')
    review = json.loads((co['secure'] / 'selected_ids.json').read_text(encoding='utf-8'))
    rb = load_bundle(co['bundle'] if 'bundle' in co else PKG / 'data/clinician_50/clinician_safe')['cases']
    per = reviewer_events(a.events)
    simulated = any(e['correction_source'] != 'clinician' for evs in per.values() for e in evs)
    src = 'SIMULATED' if simulated else 'clinician'
    sets = dict(per); sets['CONSENSUS'] = consensus_events(per)
    fut, gate_pids = batches['future_eval_1'], batches['validation_gate_1']
    v0 = CentroidHungarianMatcher(cfg['v0']['gate_mm'])
    p0 = {p: v0.predict(cases[p])['pairs'] for p in fut}
    rows, summ, gates = [], [], []
    rows += outcome_rows(cases, gt, fut, p0, 'frozen_V0', src)
    for name, evs in sets.items():
        cand = fit_candidate(evs, cases, cfg, seed)
        pc = {p: cand.predict(cases[p])['pairs'] for p in fut}
        g = run_gate(cases, gt, v0, cand, gate_pids, cfg['non_regression_thresholds'], 500, seed)
        gates.append(dict(readers=name, accepted=g['accepted'], reason=g['reason'], n_corrections=sum(map(is_correction, evs))))
        rows += outcome_rows(cases, gt, fut, pc, f'naive_update|{name}', src)
        rows += outcome_rows(cases, gt, fut, pc if g['accepted'] else p0, f'gated_coevolution|{name}', src)
    # memory only: identical to V0 on future patients by design (corrections stay in the 50 review states)
    rows += [dict(r, condition='memory_only') for r in outcome_rows(cases, gt, fut, p0, 'x', src)]
    # immediate patient-level effect on the 50 review patients
    rp = [p for p in review if p in cases]
    pv0 = {p: v0.predict(cases[p])['pairs'] for p in rp}
    rev = outcome_rows(cases, gt, rp, pv0, 'review_V0_state', src)
    for name, evs in sets.items():
        corr = {}
        for p in rp:
            dec = op.decisions([e for e in evs if e['patient_id'] == p])
            corr[p] = [tuple(x) for x in op.resolve(rb[p], dec)['pairs']] if p in rb else pv0[p]
        rev += outcome_rows(cases, gt, rp, corr, f'review_corrected_state|{name}', src)
    df = pd.DataFrame(rows); dr = pd.DataFrame(rev)
    df.to_csv(run / 'future_patient_outcomes.csv', index=False); dr.to_csv(run / 'review_patient_outcomes.csv', index=False)
    for d, base, scope in ((df, 'frozen_V0', 'future_144'), (dr, 'review_V0_state', 'review_50')):
        for cond in d.condition.unique():
            if cond == base:
                continue
            ch, out = paired(d, cond, base)
            for ep, v in out.items():
                summ.append(dict(scope=scope, condition=cond, endpoint=ep, n=d[d.condition == base].shape[0], changed_conclusions=ch, **v))
    S = pd.DataFrame(summ); S.to_csv(run / 'paired_summary.csv', index=False); pd.DataFrame(gates).to_csv(run / 'gates.csv', index=False)
    (run / 'results.json').write_text(json.dumps(dict(event_source=src, events_dir=str(a.events), n_future=len(fut), n_review=len(rp),
                                                     ref_response_future=df[df.condition == 'frozen_V0'].ref_response.value_counts().to_dict()), indent=1))
    pd.set_option('display.width', 250)
    print('event source:', src); print(pd.DataFrame(gates).to_string(index=False))
    print(S[S.endpoint.isin(['response_wrong', 'pd_wrong', 'missed_new'])].round(4).to_string(index=False)); print('done', run)


if __name__ == '__main__':
    main()
