#!/usr/bin/env python3
"""Exp 3.1 analysis (pre-specified analysis plan). A: strong baselines on the natural / late timelines (existing multi-visit raw files read-only +
new systems); B: error propagation and recovery on the revision timeline. Writes outputs/exp31/results/*.csv.   python analyze31.py"""
import json
import sys

import numpy as np
import pandas as pd

from common import MV, OUT, A, Q, cohort, reference, steps
import analyze as AN  # noqa: E402  (multi-visit scoring, read-only)

MVOUT = MV.parents[1] / 'outputs/multivisit_longitudinal'
RES = OUT / 'results'
NB, SEED = 2000, 20261011


def load_det(path, system):
    d, _ = AN.load_onco(path)
    if len(d):
        d['system'] = system
    return d


def boot_diff(a, b, n=NB, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed); k = len(a)
    bs = [np.nanmean(a[i]) - np.nanmean(b[i]) for i in (rng.integers(0, k, k) for _ in range(n))]
    return float(np.nanmean(a) - np.nanmean(b)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def part_a(coh):
    raw = MVOUT / 'raw'
    frames = [AN.load_agent(raw / f'{s}_natural.jsonl') for s in ('stateless', 'stateful', 'onco_llm')]
    frames.append(AN.load_onco(raw / 'oncostate_natural_run1.jsonl')[0])
    frames.append(AN.load_agent(OUT / 'raw/longcontext_natural.jsonl'))
    frames.append(load_det(OUT / 'raw/eventsourced_natural.jsonl', 'eventsourced'))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    sc = AN.score_frame(df, coh)
    per_rep, per_pat = AN.patient_table(sc, coh)
    per_pat = per_pat.rename(columns={'full_trajectory_consistent': 'full_trajectory_consistency'})
    rows, pairs = [], []
    M = ['longitudinal_query_accuracy', 'full_trajectory_consistency', 'history_retrieval_accuracy', 'current_state_accuracy', 'acc_Q6']
    w = {m: per_pat.pivot(index='patient_id', columns='system', values=m) for m in M}
    for s in w[M[0]].columns:
        for m in M:
            x = w[m][s].dropna().to_numpy(); rng = np.random.default_rng(SEED)
            bs = [x[i].mean() for i in (rng.integers(0, len(x), len(x)) for _ in range(NB))]
            rows.append(dict(system=s, metric=m, estimate=x.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n=len(x)))
    for other in [c for c in w[M[0]].columns if c != 'oncostate']:
        for m in M:
            j = w[m][['oncostate', other]].dropna()
            d, lo, hi = boot_diff(j.oncostate, j[other])
            pairs.append(dict(comparison=f'oncostate - {other}', metric=m, difference=d, lo=lo, hi=hi, n=len(j)))
    depth = []
    for (s, p, r), g in sc.groupby(['system', 'patient_id', 'repeat_id']):
        for dpt in sorted(g.visit_depth.unique()):
            depth.append(dict(system=s, patient_id=p, repeat_id=r, visit_depth=dpt, consistent=float(AN.consistency(g, upto=dpt)['consistent'])))
    depth = pd.DataFrame(depth).groupby(['system', 'patient_id', 'visit_depth']).consistent.mean().reset_index()
    depth = depth.groupby(['system', 'visit_depth']).consistent.agg(['mean', 'count']).reset_index()
    pd.DataFrame(rows).to_csv(RES / 'A_aggregate.csv', index=False); pd.DataFrame(pairs).to_csv(RES / 'A_paired.csv', index=False)
    depth.to_csv(RES / 'A_consistency_by_depth.csv', index=False); per_pat.to_csv(RES / 'A_per_patient.csv', index=False)
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def score_mode(df, coh, mode):
    out = []
    for r in df.itertuples():
        v = coh[r.patient_id]
        ref = reference(r.patient_id, v, list(r.available), r.current_visit, mode).get(r.query)
        out.append(dict(evaluable=ref is not None, correct=bool(ref is not None and Q.score(r.query, r.answer, ref)), reference=ref))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def part_late(coh):
    raw = MVOUT / 'raw'
    frames = [AN.load_agent(raw / f'{s}_late.jsonl') for s in ('stateful',)]
    frames.append(AN.load_onco(raw / 'oncostate_late_run1.jsonl')[0])
    frames.append(AN.load_agent(OUT / 'raw/longcontext_late.jsonl'))
    frames.append(load_det(OUT / 'raw/eventsourced_late.jsonl', 'eventsourced'))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    sc = score_mode(df, coh, 'late')
    rows = []
    for (s, p, r), g in sc.groupby(['system', 'patient_id', 'repeat_id']):
        ev = g[g.evaluable]
        after = ev[ev.step_index >= 2]
        rows.append(dict(system=s, patient_id=p, repeat_id=r, inserted_correctly=float(ev[(ev.step_index == 2) & ev['query'].isin(['Q2', 'Q8'])].correct.all()),
                         later_all_correct=float(after.correct.all()), accuracy_after_arrival=after.correct.mean()))
    t = pd.DataFrame(rows).groupby('system').mean(numeric_only=True).reset_index()
    t.to_csv(RES / 'A_late_arrival.csv', index=False)
    return t


def part_b(coh):
    raw = OUT / 'raw'
    frames = [load_det(raw / f'{s}_revision.jsonl', s) for s in ('oncostate', 'nondurable', 'eventsourced')]
    frames += [AN.load_agent(raw / f'{s}_revision.jsonl') for s in ('stateless', 'stateful', 'longcontext')]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    sc = score_mode(df, coh, 'revision')
    rows = []
    for (s, p, r), g in sc.groupby(['system', 'patient_id', 'repeat_id']):
        v = coh[p]
        rec2 = A.measurement(p, v[1])[0]['burden_ml']; bad = round(rec2 * 1.6, 2)
        ev = g[g.evaluable]
        post = ev[ev.step_index >= 2]
        steps_post = sorted(post.step_index.unique())
        ok_by_step = {k: post[post.step_index == k].correct.all() for k in steps_post}
        ttc = next((i for i, k in enumerate(steps_post) if all(ok_by_step[j] for j in steps_post[i:])), np.nan)
        stale = any(Q._num(x) and abs(x - bad) < .05 for a in post.answer if a is not None
                    for x in ([a.get('previous_burden_ml'), a.get('first_burden_ml'), a.get('nadir_burden_ml')] +
                              [h.get('burden_ml') for h in a.get('history', []) if str(h.get('visit')) == v[1]]))
        pre = ev[(ev.step_index == 1) & (ev['query'] == 'Q6')]
        rows.append(dict(system=s, patient_id=p, repeat_id=r, downstream_error_rate=1 - post.correct.mean(),
                         progression_call_error=1 - post[post['query'] == 'Q6'].correct.mean(),
                         recovered_at_revision=float(ok_by_step.get(steps_post[0], False)), visits_to_correction=ttc,
                         never_recovered=float(np.isnan(ttc)), stale_value_used=float(stale),
                         false_step_at_v2=float(pre.answer.map(lambda a: (a or {}).get('current_step')).iloc[0] != Q.step(A.measurement(p, v[0])[0]['burden_ml'], rec2))
                         if len(pre) else np.nan))
    pp = pd.DataFrame(rows).groupby(['system', 'patient_id']).mean(numeric_only=True).reset_index()
    agg = []
    for s, g in pp.groupby('system'):
        for m in ('downstream_error_rate', 'progression_call_error', 'recovered_at_revision', 'never_recovered', 'stale_value_used', 'false_step_at_v2'):
            x = g[m].dropna().to_numpy(); rng = np.random.default_rng(SEED)
            bs = [x[i].mean() for i in (rng.integers(0, len(x), len(x)) for _ in range(NB))] if len(x) else [np.nan]
            agg.append(dict(system=s, metric=m, estimate=x.mean() if len(x) else np.nan, lo=np.nanquantile(bs, .025), hi=np.nanquantile(bs, .975), n=len(x)))
        agg.append(dict(system=s, metric='median_visits_to_correction', estimate=g.visits_to_correction.median(), n=int(g.visits_to_correction.notna().sum())))
    pd.DataFrame(agg).to_csv(RES / 'B_aggregate.csv', index=False); pp.to_csv(RES / 'B_per_patient.csv', index=False)
    return pd.DataFrame(agg)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    A.load_measure_cache(MVOUT / 'measurements_frozen.json')
    pd.set_option('display.width', 250)
    a, p = part_a(cohort(0)); print(a[a.metric.isin(['longitudinal_query_accuracy', 'full_trajectory_consistency'])].round(3).to_string(index=False))
    print(p[p.metric.isin(['longitudinal_query_accuracy', 'full_trajectory_consistency'])].round(3).to_string(index=False))
    print(part_late(cohort(4)).round(3).to_string(index=False))
    print(part_b(cohort(4)).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
