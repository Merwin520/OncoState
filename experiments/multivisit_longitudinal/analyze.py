#!/usr/bin/env python3
"""Offline evaluation (runs only after all system outputs are frozen). The ONLY code that uses full patient trajectories.

Reference answers per step are recomputed from the frozen measurements for the visits AVAILABLE at that step (natural mode: identical
to reference_trajectories.csv, asserted). Writes per_visit_results.csv, per_patient_trajectory_results.csv, aggregate_metrics.csv,
visit_depth_metrics.csv, efficiency_metrics.csv, paired_bootstrap_statistics.csv, late_arrival_stress_test.csv,
treatment_chronology_results.csv, summary_table.csv.     python analyze.py [--pilot]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mu_glioma_adapter as A  # noqa: E402
import queries as Q  # noqa: E402
import timeline as T  # noqa: E402

NB, SEED = 5000, 20261006
LABEL = {'stateless': 'Stateless Tool Agent', 'stateful': 'Generic Stateful Tool Agent', 'oncostate': 'OncoState',
         'onco_llm': 'OncoState state + LLM reader (exploratory)'}
PRIMARY = ['stateless', 'stateful', 'oncostate']


# ----------------------------------------------------------------------------------------------------------------- loading
def load_agent(path):
    """Latest complete sequence per (patient, repeat); infrastructure (API) failures never occurred silently: they are re-run."""
    if not path.exists():
        return pd.DataFrame()
    seqs, ends = {}, {}
    for l in path.read_text().splitlines():
        j = json.loads(l)
        if j['query'] == '_end':
            ends[(j['patient_id'], j['repeat_id'])] = j['sequence_uid']
        else:
            seqs.setdefault(j['sequence_uid'], []).append(j)
    rows = []
    for (pid, rep), uid in ends.items():
        for e in seqs[uid]:
            if e['query'].startswith('_'):
                continue
            fr = e.get('failure_reason') or ''
            if 'maximum context length' in fr:
                fr = 'context_length_exceeded'                   # agent failure (counted), as in Exp 1
            calls = e['tool_calls']
            rows.append(dict(system=e['system'], mode=e['mode'], patient_id=pid, repeat_id=rep, step_index=e['step_index'], current_visit=e['current_visit'],
                             available=e['available'], tag=e['tag'], visit_depth=e['visit_depth'], query=e['query'], answer=e['structured_answer'],
                             success=e['success'], failure_reason=fr or None, n_tool_calls=len(calls),
                             source_reads=[c['visit_read'] for c in calls if c.get('visit_read')],
                             list_calls=sum(c['name'] == 'list_visits' for c in calls), calc_calls=sum(c['name'] == 'calculate' for c in calls),
                             future_blocked=sum(bool(c.get('future_blocked')) for c in calls),
                             uncached_compute_sec=sum(c.get('compute_seconds_if_uncached', 0) for c in calls),
                             llm_calls=e['n_turns'], tokens=e['token_count'], latency_sec=e['latency_sec'],
                             memory_bytes_in=e.get('memory_bytes_in'), memory_visits_after=e.get('memory_visits_after'),
                             future_successful_reads=sum(1 for x in e['clock_log'] if x['allowed'] and x['visit'] not in e['available'])))
    return pd.DataFrame(rows)


def load_onco(path):
    rows, syncs = [], []
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    for l in path.read_text().splitlines():
        j = json.loads(l)
        if j['query'] == '_sync':
            syncs.append(j)
        else:
            rows.append(dict(system='oncostate', mode=j['mode'], patient_id=j['patient_id'], repeat_id=0, step_index=j['step_index'],
                             current_visit=j['current_visit'], available=j['available'], tag=j['tag'], visit_depth=j['visit_depth'], query=j['query'],
                             answer=j['structured_answer'], success=j['success'], failure_reason=j['failure_reason']))
    return pd.DataFrame(rows), pd.DataFrame(syncs)


# ----------------------------------------------------------------------------------------------------------------- scoring
_ref_cache = {}


def ref_answers(pid, visits, avail):
    key = (pid, tuple(avail))
    if key not in _ref_cache:
        recs = {v: A.visit_record(pid, v, visits)[0] for v in avail}
        _ref_cache[key] = Q.reference_answers([(v, recs[v]['measurements']['burden_ml']) for v in avail],
                                              {v: recs[v]['treatments_reported_since_previous_visit'] for v in avail})
    return _ref_cache[key]


def mentions_future(ans, pid, visits, avail):
    ids = set(re.findall(r'Timepoint_\d+', json.dumps(ans))) if ans is not None else set()
    return bool(ids & (set(visits) - set(avail)))


def score_frame(df, coh):
    out = []
    for r in df.itertuples():
        ref = ref_answers(r.patient_id, coh[r.patient_id], r.available)
        rq = ref.get(r.query)
        out.append(dict(reference=rq, evaluable=rq is not None, correct=bool(rq is not None and Q.score(r.query, r.answer, rq)),
                        future_mention=mentions_future(r.answer, r.patient_id, coh[r.patient_id], r.available)))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# ----------------------------------------------------------------------------------------------------------------- consistency
def consistency(g, upto=None):
    """Full-trajectory consistency components for ONE (system, patient, repeat) natural sequence, through visit depth `upto`."""
    g = g[g.visit_depth <= upto] if upto else g
    A_ = {(r.step_index, r.query): r for r in g.itertuples()}
    steps = sorted(g.step_index.unique())
    own_q1, c = {}, dict(C1=True, C2=True, C3=True, C4=True, C5=True)
    for s in steps:
        q1 = A_.get((s, 'Q1'))
        cur = q1.current_visit if q1 is not None else None
        get = lambda q: A_[(s, q)].answer if (s, q) in A_ and A_[(s, q)].success else None  # noqa: E731
        for q in ('Q2', 'Q8'):
            if (s, q) in A_ and not A_[(s, q)].correct:
                c['C1'] = False
        reported = []
        if get('Q2'):
            reported.append((str(get('Q2')['previous_visit']), get('Q2')['previous_burden_ml']))
        if get('Q4'):
            reported.append((str(get('Q4')['first_visit']), get('Q4')['first_burden_ml']))
        if get('Q5') and str(get('Q5')['nadir_visit']) != cur:
            reported.append((str(get('Q5')['nadir_visit']), get('Q5')['nadir_burden_ml']))
        if get('Q8'):
            reported += [(str(h['visit']), h['burden_ml']) for h in get('Q8')['history'] if str(h['visit']) != cur]
        for v, val in reported:
            if v in own_q1 and not Q.close_ml(val, own_q1[v]):
                c['C2'] = False
        b = get('Q1')['burden_ml'] if get('Q1') else None
        if b is not None:
            if get('Q2') and get('Q3') and not Q.close_ml(get('Q3')['delta_ml'], round(b - get('Q2')['previous_burden_ml'], 2)):
                c['C3'] = False
            if get('Q4') and not Q.close_ml(get('Q4')['delta_ml'], round(b - get('Q4')['first_burden_ml'], 2)):
                c['C3'] = False
            if get('Q5') and not Q.close_ml(get('Q5')['current_minus_nadir_ml'], round(b - get('Q5')['nadir_burden_ml'], 2)):
                c['C3'] = False
        if any(A_[(s, q)].future_mention for (ss, q) in A_ if ss == s):
            c['C4'] = False
        if any(not A_[(s, q)].success for (ss, q) in A_ if ss == s):
            c['C5'] = False
        if q1 is not None and q1.success:
            own_q1[cur] = q1.answer['burden_ml']
    c['consistent'] = all(c.values())
    return c


# ----------------------------------------------------------------------------------------------------------------- statistics
def boot_ci(x, idx):
    x = np.asarray(x, float)
    b = np.nanmean(x[idx], axis=1)
    return float(np.nanmean(x)), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def patient_table(sc, coh):
    """Per (system, patient): metrics averaged over repeats (expected value under repeated sampling, as in Exp 1)."""
    rows = []
    for (sysm, pid, rep), g in sc.groupby(['system', 'patient_id', 'repeat_id']):
        ev = g[g.evaluable]
        lon = ev[ev['query'].isin(Q.LONGITUDINAL)]
        his = ev[ev['query'].isin(Q.HISTORY_RETRIEVAL)]
        cur = ev[ev['query'] == 'Q1']
        c = consistency(g)
        rows.append(dict(system=sysm, patient_id=pid, repeat_id=rep, n_visits=len(coh[pid]), longitudinal_query_accuracy=lon.correct.mean(),
                         history_retrieval_accuracy=his.correct.mean(), current_state_accuracy=cur.correct.mean(),
                         all_query_accuracy=ev.correct.mean(), treatment_chronology_accuracy=ev[ev['query'] == 'Q7'].correct.mean(),
                         valid_answer_rate=g.success.mean(), full_trajectory_consistent=float(c['consistent']),
                         **{f'consistency_{k}': float(v) for k, v in c.items() if k != 'consistent'},
                         **{f'acc_{q}': ev[ev['query'] == q].correct.mean() for q in Q.QIDS},
                         n_queries=len(ev), n_failed=int((~g.success).sum()), future_mentions=int(g.future_mention.sum())))
    per_rep = pd.DataFrame(rows)
    per_pat = per_rep.groupby(['system', 'patient_id']).mean(numeric_only=True).reset_index()
    return per_rep, per_pat


METRICS = ['longitudinal_query_accuracy', 'full_trajectory_consistency', 'history_retrieval_accuracy', 'current_state_accuracy',
           'treatment_chronology_accuracy', 'all_query_accuracy', 'valid_answer_rate', 'consistency_C1', 'consistency_C2', 'consistency_C3',
           'consistency_C4', 'consistency_C5'] + [f'acc_{q}' for q in Q.QIDS]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); ap.add_argument('--prefix', default=''); a = ap.parse_args()
    P = a.prefix
    OUT = T.OUT / ('pilot' if a.pilot else '')
    RAW = OUT / 'raw'
    A.load_measure_cache(T.OUT / 'measurements_frozen.json')
    coh = T.cohort()
    log = []

    # ---- natural mode
    frames, syncs = [], None
    for s in ('stateless', 'stateful', 'onco_llm'):
        frames.append(load_agent(RAW / f'{P}{s}_natural.jsonl'))
    od, syncs = load_onco(RAW / 'oncostate_natural_run1.jsonl')
    frames.append(od)
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    sc = score_frame(df, coh)
    # natural-mode references == frozen reference_trajectories.csv
    rt = pd.read_csv(T.OUT / 'reference_trajectories.csv')
    for r in rt.itertuples():
        v = coh[r.patient_id]
        assert Q.dumps(ref_answers(r.patient_id, v, v[:r.visit_depth])) == r.reference_answers, r
    log.append('reference recomputation == reference_trajectories.csv for all visits')
    pv = sc.copy()
    pv['answer'] = pv.answer.map(lambda x: json.dumps(x) if x is not None else None)
    pv['reference'] = pv.reference.map(lambda x: json.dumps(x) if x is not None else None)
    pv['n_source_reads'] = pv.source_reads.map(lambda x: len(x) if isinstance(x, list) else np.nan)
    pv.drop(columns=['source_reads', 'memory_visits_after', 'available']).to_csv(OUT / 'per_visit_results.csv', index=False)

    per_rep, per_pat = patient_table(sc, coh)
    per_pat = per_pat.rename(columns={'full_trajectory_consistent': 'full_trajectory_consistency'})
    per_rep.to_csv(OUT / 'per_patient_trajectory_results_by_repeat.csv', index=False)
    per_pat.to_csv(OUT / 'per_patient_trajectory_results.csv', index=False)

    systems = [s for s in ['stateless', 'stateful', 'oncostate', 'onco_llm'] if s in set(per_pat.system)]
    pats = sorted(coh)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(pats), size=(NB, len(pats)))
    W = {s: per_pat[per_pat.system == s].set_index('patient_id').reindex(pats) for s in systems}
    agg = []
    for s in systems:
        for m in METRICS:
            est, lo, hi = boot_ci(W[s][m].values, idx)
            agg.append(dict(system=s, method=LABEL[s], metric=m, estimate=est, ci95_lo=lo, ci95_hi=hi, n_patients=int(W[s][m].notna().sum())))
        if s in ('stateless', 'stateful', 'onco_llm'):
            r = per_rep[per_rep.system == s].groupby('repeat_id')[['longitudinal_query_accuracy', 'full_trajectory_consistent']].mean()
            agg.append(dict(system=s, method=LABEL[s], metric='longitudinal_query_accuracy_repeat_sd', estimate=float(r.iloc[:, 0].std(ddof=1)) if len(r) > 1 else np.nan))
            agg.append(dict(system=s, method=LABEL[s], metric='full_trajectory_consistency_repeat_sd', estimate=float(r.iloc[:, 1].std(ddof=1)) if len(r) > 1 else np.nan))
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT / 'aggregate_metrics.csv', index=False)

    pairs = [('oncostate', 'stateful'), ('stateful', 'stateless'), ('oncostate', 'stateless')]
    if 'onco_llm' in systems:
        pairs += [('onco_llm', 'stateful')]
    pb = []
    for x, y in pairs:
        if x not in systems or y not in systems:
            continue
        for m in METRICS:
            d = W[x][m].values - W[y][m].values
            b = np.nanmean(d[idx], axis=1)
            pb.append(dict(comparison=f'{LABEL[x]} - {LABEL[y]}', primary=(x, y) in pairs[:2], metric=m, difference=float(np.nanmean(d)),
                           ci95_lo=float(np.nanpercentile(b, 2.5)), ci95_hi=float(np.nanpercentile(b, 97.5)),
                           ci97_5_bonferroni_lo=float(np.nanpercentile(b, 1.25)), ci97_5_bonferroni_hi=float(np.nanpercentile(b, 98.75)),
                           n_patients=int(np.sum(~np.isnan(d)))))
    pd.DataFrame(pb).to_csv(OUT / 'paired_bootstrap_statistics.csv', index=False)

    # ---- visit depth
    dep = []
    for d in range(1, 7):
        dp = [p for p in pats if len(coh[p]) >= d]
        if not dp:
            continue
        idd = np.random.default_rng(SEED + d).integers(0, len(dp), size=(NB, len(dp)))
        at = sc[sc.visit_depth == d]
        for s in systems:
            g = at[(at.system == s) & at.evaluable]
            lon = g[g['query'].isin(Q.LONGITUDINAL)].groupby(['patient_id', 'repeat_id']).correct.mean().groupby('patient_id').mean().reindex(dp)
            cur = g[g['query'] == 'Q1'].groupby(['patient_id', 'repeat_id']).correct.mean().groupby('patient_id').mean().reindex(dp)
            cr = [(pid, float(consistency(gg, upto=d)['consistent']))
                  for (pid, _), gg in sc[(sc.system == s) & sc.patient_id.isin(dp)].groupby(['patient_id', 'repeat_id'])]
            cum = pd.DataFrame(cr, columns=['patient_id', 'c']).groupby('patient_id').c.mean().reindex(dp) if cr else pd.Series(np.nan, index=dp)
            row = dict(system=s, method=LABEL[s], visit_depth=d, n_patients=len(dp))
            for name, ser in (('longitudinal_accuracy', lon), ('current_state_accuracy', cur), ('cumulative_trajectory_consistency', cum)):
                if ser.notna().any():
                    est, lo, hi = boot_ci(ser.values, idd)
                    row.update({name: est, f'{name}_ci_lo': lo, f'{name}_ci_hi': hi})
            if 'longitudinal_accuracy' in row:
                row['longitudinal_error'] = 1 - row['longitudinal_accuracy']
                row['longitudinal_error_ci_lo'], row['longitudinal_error_ci_hi'] = 1 - row['longitudinal_accuracy_ci_hi'], 1 - row['longitudinal_accuracy_ci_lo']
            dep.append(row)
    dep = pd.DataFrame(dep)
    dep.to_csv(OUT / 'visit_depth_metrics.csv', index=False)

    # ---- efficiency per new visit (natural mode)
    eff = []
    for s in systems:
        if s == 'oncostate':
            for r in syncs.itertuples():
                eff.append(dict(system=s, patient_id=r.patient_id, repeat_id=0, visit_depth=r.visit_depth, source_reads=len(r.visits_read),
                                redundant_source_reads=0, history_reconstructions=0, list_calls=0, tool_calls=r.tool_calls, llm_calls=0, tokens=0,
                                latency_sec=r.sync_sec + r.query_sec, uncached_compute_sec=r.uncached_compute_sec, state_loads=r.state_reads))
            continue
        g = sc[sc.system == s].sort_values(['patient_id', 'repeat_id', 'step_index'])
        for (pid, rep), gg in g.groupby(['patient_id', 'repeat_id']):
            seen = set()
            for st, gs in gg.groupby('step_index', sort=True):
                reads = [v for lst in gs.source_reads for v in lst]
                cur = gs.current_visit.iloc[0]
                red = 0
                for v in reads:
                    red += v in seen
                    seen.add(v)
                recon = sum(any(v != cur for v in lst) for lst in gs.source_reads)
                eff.append(dict(system=s, patient_id=pid, repeat_id=rep, visit_depth=int(gs.visit_depth.iloc[0]), source_reads=len(reads),
                                redundant_source_reads=red, history_reconstructions=recon, list_calls=int(gs.list_calls.sum()),
                                tool_calls=int(gs.n_tool_calls.sum()), llm_calls=int(gs.llm_calls.sum()), tokens=int(gs.tokens.sum()),
                                latency_sec=float(gs.latency_sec.sum()), uncached_compute_sec=float(gs.uncached_compute_sec.sum()),
                                state_loads=1 if s == 'stateful' else 0))
    eff = pd.DataFrame(eff)
    effcols = ['source_reads', 'redundant_source_reads', 'history_reconstructions', 'list_calls', 'tool_calls', 'llm_calls', 'tokens', 'latency_sec',
               'uncached_compute_sec', 'state_loads']
    effp = eff.groupby(['system', 'patient_id']).mean(numeric_only=True).reset_index()
    erows = []
    for s in systems:
        w = effp[effp.system == s].set_index('patient_id').reindex(pats)
        for m in effcols:
            est, lo, hi = boot_ci(w[m].values, idx)
            erows.append(dict(system=s, method=LABEL[s], metric=f'{m}_per_new_visit', estimate=est, ci95_lo=lo, ci95_hi=hi, depth='all'))
        for d, gd in eff[eff.system == s].groupby('visit_depth'):
            for m in ('source_reads', 'redundant_source_reads', 'tool_calls', 'llm_calls', 'latency_sec', 'tokens'):
                erows.append(dict(system=s, method=LABEL[s], metric=f'{m}_per_new_visit', estimate=float(gd[m].mean()), depth=int(d),
                                  n_patients=int(gd.patient_id.nunique())))
    pd.DataFrame(erows).to_csv(OUT / 'efficiency_metrics.csv', index=False)

    # ---- treatment chronology (Q7)
    q7 = sc[(sc['query'] == 'Q7') & sc.evaluable]
    tc = []
    for s in systems:
        g = q7[q7.system == s]
        ser = g.groupby('patient_id').correct.mean().reindex(pats)
        est, lo, hi = boot_ci(ser.values, idx)
        tc.append(dict(system=s, method=LABEL[s], visit_depth='all', q7_accuracy=est, ci95_lo=lo, ci95_hi=hi, n_patients=int(ser.notna().sum()), n_queries=len(g)))
        for d, gd in g.groupby('visit_depth'):
            tc.append(dict(system=s, method=LABEL[s], visit_depth=int(d), q7_accuracy=float(gd.groupby('patient_id').correct.mean().mean()),
                           n_patients=int(gd.patient_id.nunique()), n_queries=len(gd)))
    pd.DataFrame(tc).to_csv(OUT / 'treatment_chronology_results.csv', index=False)

    # ---- late arrival (secondary)
    lat = []
    ls_ = load_agent(RAW / f'{P}stateful_late.jsonl')
    lo_, lsy = load_onco(RAW / 'oncostate_late_run1.jsonl')
    late = pd.concat([x for x in (ls_, lo_) if len(x)], ignore_index=True) if (len(ls_) or len(lo_)) else pd.DataFrame()
    if len(late):
        late = score_frame(late, coh)
        for (s, pid, rep), g in late.groupby(['system', 'patient_id', 'repeat_id']):
            v = coh[pid]
            ans = {(r.step_index, r.query): r for r in g.itertuples()}
            ok = lambda st, q: bool((st, q) in ans and ans[(st, q)].correct)  # noqa: E731
            h1 = ans.get((1, 'Q8')); h2 = ans.get((2, 'Q8'))
            unchanged = False
            if h1 is not None and h2 is not None and h1.success and h2.success:
                b1 = {str(x['visit']): x['burden_ml'] for x in h1.answer['history']}
                b2 = {str(x['visit']): x['burden_ml'] for x in h2.answer['history']}
                unchanged = all(k in b2 and Q.close_ml(b2[k], b1[k]) for k in (v[0], v[2]) if k in b1) and v[0] in b1 and v[2] in b1
            later = g[g.step_index >= 3]
            row = dict(system=s, method=LABEL[s], patient_id=pid, repeat_id=rep, n_visits=len(v), hidden_visit=v[1],
                       missing_represented_before_arrival=ok(1, 'Q8') and ok(1, 'Q2'),
                       visit_inserted_chronologically=ok(2, 'Q8') and ok(2, 'Q2'),
                       unrelated_history_unchanged=bool(unchanged),
                       later_state_accuracy=float(later[later.evaluable].correct.mean()) if len(later) else np.nan,
                       later_state_all_correct=bool(later[later.evaluable].correct.all()) if len(later) else np.nan,
                       accuracy_before_arrival=float(g[(g.step_index == 1) & g.evaluable].correct.mean()),
                       accuracy_at_arrival=float(g[(g.step_index == 2) & g.evaluable].correct.mean()),
                       future_leakage=bool(g.future_mention.any()) or bool(g.get('future_successful_reads', pd.Series([0])).fillna(0).sum() > 0))
            if s == 'oncostate':
                sy = lsy[lsy.patient_id == pid].set_index('step_index')
                row['state_status_hidden_before_arrival'] = sy.loc[1, 'observation_status'].get(v[1]) == 'hidden_stress_test'
                row['state_status_late_arrival_after'] = sy.loc[2, 'observation_status'].get(v[1]) == 'late_arrival'
                row['state_no_imputation_before_arrival'] = v[1] not in sy.loc[1, 'history_keys']
            lat.append(row)
    lat = pd.DataFrame(lat)
    lat.to_csv(OUT / 'late_arrival_stress_test.csv', index=False)
    if len(lat):
        lsum = lat.groupby('method').mean(numeric_only=True).reset_index()
        lsum['n_patients'] = lat.groupby('method').patient_id.nunique().values
        lsum.to_csv(OUT / 'late_arrival_summary.csv', index=False)

    # ---- summary table
    e_all = pd.DataFrame(erows)
    e_all = e_all[e_all.depth == 'all']
    fmt = lambda s, m: (lambda r: f'{r.estimate:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]')(agg[(agg.system == s) & (agg.metric == m)].iloc[0])  # noqa: E731
    fe = lambda s, m, k=2: f"{e_all[(e_all.system == s) & (e_all.metric == m)].estimate.iloc[0]:.{k}f}"  # noqa: E731
    tab = []
    for s in systems:
        tab.append({'Method': LABEL[s], 'Longitudinal Query Accuracy': fmt(s, 'longitudinal_query_accuracy'),
                    'Full-Trajectory Consistency': fmt(s, 'full_trajectory_consistency'),
                    'History Retrieval Accuracy': fmt(s, 'history_retrieval_accuracy'), 'Current-State Accuracy': fmt(s, 'current_state_accuracy'),
                    'Source Reads / new visit': fe(s, 'source_reads_per_new_visit'), 'Tool Calls / new visit': fe(s, 'tool_calls_per_new_visit'),
                    'LLM Calls / new visit': fe(s, 'llm_calls_per_new_visit'),
                    'Execution Time / new visit (s)': fe(s, 'latency_sec_per_new_visit', 3 if s == 'oncostate' else 1)})
    pd.DataFrame(tab).to_csv(OUT / 'summary_table.csv', index=False)
    (OUT / 'analysis_log.txt').write_text('\n'.join(log) + '\n')
    print(pd.DataFrame(tab).to_string())


if __name__ == '__main__':
    main()
