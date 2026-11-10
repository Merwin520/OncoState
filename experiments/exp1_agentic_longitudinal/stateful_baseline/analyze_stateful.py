#!/usr/bin/env python3
"""Controlled comparison. Both agents run on the same local model (Qwen3.5-9B via vLLM, see experiment_config.yaml); the archived
GPT-5.2 stateless run is not mixed in. Four-way comparison (Direct VLM | stateless tool agent | generic stateful tool agent | OncoState) with the UNCHANGED Exp 2.2 evaluation:
per-patient sufficient statistics (exp22b_common.evaluate via exp22b_run.agent_patient_ev), patient-level bootstrap (exp22b_stats.analyse /
paired), efficiency (exp22b_run.efficiency), cross-query consistency (exp22b_run.consistency) and correction retention
(exp22b_retention.reflects on the same 60 correction events). Writes outputs/stateful_baseline/*.csv.
  python analyze_stateful.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exp22b_run as RUN  # noqa: E402
import exp22b_stats as S  # noqa: E402
import exp22b_retention as RET  # noqa: E402

OUT = HERE.parents[2] / 'outputs/stateful_baseline'
EXT = RUN.BASE.parent / 'exp22_external_final'
STATELESS, STATEFUL, ONCO = RUN.AGENT, 'Generic stateful agent', 'Full OncoAgent'
LABEL = {ONCO: 'OncoState', STATELESS: 'Stateless tool agent', STATEFUL: 'Generic stateful tool agent'}


def det_cached():
    f = OUT / 'det_cache.pkl'
    if f.exists():
        return pickle.loads(f.read_bytes())
    det, bad = RUN.deterministic()
    f.write_bytes(pickle.dumps((det, bad)))
    return det, bad


def load_raw(path):
    latest = {}
    if not path.exists():
        return []
    for l in path.read_text().splitlines():
        j = json.loads(l)
        if RUN.is_infra_failure(j) and 'maximum context length' not in (j.get('failure_reason') or ''):
            continue                                            # true infrastructure errors only (none occurred in this run)
        if 'maximum context length' in (j.get('failure_reason') or ''):
            j = {**j, 'failure_reason': 'context_length_exceeded'}   # an agent failure (prompt + memory too long), counted as failed
        latest[(j['patient_id'], j['repeat_id'], j['query_id'], j.get('run_tag'))] = j
    return list(latest.values())


def per_patient(det, recs, patients):
    by = {}
    for r in recs:
        by.setdefault((r['patient_id'], r['repeat_id']), {})[r['query_id']] = r
    vecs, sc, cons, bok = ({p: [] for p in patients} for _ in range(4))
    for (p, r), qs in by.items():
        if p not in vecs or len(qs) < 4:
            continue
        ev, pred = RUN.agent_patient_ev(det[(p, ONCO)]['ref'], qs)
        vecs[p].append(S.vec(ev)); sc[p].append(float(all(qs[q]['success'] for q in qs))); cons[p].append(RUN.consistency(qs, pred))
        bok[p].append(ev['burden_ok'])
    missing = [p for p in patients if not vecs[p]]
    return by, vecs, sc, cons, bok, missing


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det, bad = det_cached()
    patients = [p for p in RUN.LOCKED if all((p, m) in det for m in (ONCO, 'Reference-mask oracle', 'Conventional (centroid)', 'Conventional (IoU)'))]
    sl = load_raw(OUT / 'stateless_raw.jsonl')                     # stateless agent re-run with the SAME local model as the stateful agent
    sf = load_raw(OUT / 'generic_stateful_raw.jsonl')
    per, succ, info = {}, {}, {}
    per[ONCO] = {p: S.vec(det[(p, ONCO)]['ev']) for p in patients}; succ[ONCO] = {p: 1.0 for p in patients}
    for name, recs in ((STATELESS, sl), (STATEFUL, sf)):
        by, vecs, sc, cons, bok, missing = per_patient(det, recs, patients)
        info[name] = dict(by=by, cons=cons, bok=bok, missing=missing, reps=sorted({k[1] for k in by}))
        if missing:
            print(f'{name}: {len(missing)} patients without a complete Q1-Q4 run -> not compared (fix before reporting)')
            continue
        per[name] = {p: np.mean(vecs[p], 0) for p in patients}; succ[name] = {p: float(np.mean(sc[p])) for p in patients}
    df, point, boot, idx, Mx, Sc = S.analyse(per, succ, patients)
    df['method'] = df.method.map(lambda m: LABEL.get(m, m))
    pdiff = S.paired(point, boot)                                             # differences vs Full OncoAgent, same paired patient bootstrap
    pdiff['comparison'] = pdiff.comparison.astype(str).str.replace(ONCO, 'OncoState').str.replace(STATELESS, LABEL[STATELESS]).str.replace(STATEFUL, LABEL[STATEFUL])
    # per-patient output
    rows = []
    for m in per:
        for p in patients:
            v = S.unpack(np.asarray(per[m][p])[None])
            rows.append(dict(method=LABEL[m], patient_id=p, tp=float(v['tp'][0]), pred_pairs=float(v['pp'][0]), gt_persistent=float(v['gp'][0]),
                             cross_query_consistency=float(np.nanmean(info[m]['cons'][p])) if m in info else 1.0,
                             burden_correct=float(np.mean([x for x in info[m]['bok'][p] if x is not None])) if m in info and any(x is not None for x in info[m]['bok'][p]) else
                             (float(det[(p, ONCO)]['ev']['burden_ok']) if m == ONCO else np.nan)))
    pd.DataFrame(rows).to_csv(OUT / 'per_patient_results.csv', index=False)
    df.to_csv(OUT / 'aggregate_metrics.csv', index=False)
    pdiff.to_csv(OUT / 'bootstrap_statistics.csv', index=False)
    # efficiency: identical accounting for both agents (exp22b_run.efficiency)
    eff = []
    for name in (STATELESS, STATEFUL):
        if name not in per:
            continue
        e, _ = RUN.efficiency(det, dict(patients=patients, agent=dict(by=info[name]['by'])))
        e.loc[e.method == RUN.AGENT, 'method'] = LABEL[name]
        if name == STATEFUL:
            e.loc[e.method == LABEL[name], 'cross_query_reuse'] = 'generic persistent memory across the 4 queries'
            e = e[e.method == LABEL[name]]
        eff.append(e)
    eff = pd.concat(eff, ignore_index=True); eff['method'] = eff.method.map(lambda m: LABEL.get(m, m))
    eff.to_csv(OUT / 'efficiency_metrics.csv', index=False)
    # consistency (objective, from answers only)
    con = []
    for name in (STATELESS, STATEFUL):
        if name not in per:
            continue
        dup = []
        for (p, r), qs in info[name]['by'].items():
            q1 = qs.get('Q1')
            if q1 and q1['success']:
                m = q1['structured_answer']['matches']; b = [x['baseline_lesion_id'] for x in m]; f = [x['followup_lesion_id'] for x in m]
                dup.append(float(len(b) != len(set(b)) or len(f) != len(set(f))))
        con.append(dict(method=LABEL[name], cross_query_consistency=float(np.nanmean([np.nanmean(v) for v in info[name]['cons'].values() if v])),
                        patients_with_any_inconsistency=float(np.mean([np.nanmean(v) < .999 for v in info[name]['cons'].values() if v])),
                        runs_with_duplicate_lesion_identity=float(np.mean(dup)) if dup else np.nan))
    con.append(dict(method='OncoState', cross_query_consistency=1.0, patients_with_any_inconsistency=0.0, runs_with_duplicate_lesion_identity=0.0,
                    note='all queries read one state version (by construction)'))
    pd.DataFrame(con).to_csv(OUT / 'consistency_metrics.csv', index=False)
    # correction retention on the same 60 events
    cr = []
    ex = pd.read_csv(RUN.BASE.parent / 'exp22_external_final/edit_retention.csv')
    for r in ex[ex.system == ONCO].itertuples():
        cr.append(dict(method='OncoState', patient_id=r.patient_id, pb=r.pb, pf=r.pf, query=r.query, reflects_correction=bool(r.reflects_correction)))
    for name, path in ((STATELESS, OUT / 'stateless_correction_raw.jsonl'), (STATEFUL, OUT / 'generic_stateful_correction_raw.jsonl')):
        for r in load_raw(path):
            if r['query_id'] not in ('Q1', 'Q2', 'Q3'):
                continue
            e = json.loads(r['run_tag']) if isinstance(r.get('run_tag'), str) and r['run_tag'].startswith('{') else None
            pb, pf = (e['pb'], e['pf']) if e else (r.get('pb'), r.get('pf'))
            cr.append(dict(method=LABEL[name], patient_id=r['patient_id'], pb=pb, pf=pf, query=r['query_id'],
                           reflects_correction=bool(r['success'] and RET.reflects(r['query_id'], r['structured_answer'], pb, pf))))
    cr = pd.DataFrame(cr); cr.to_csv(OUT / 'correction_persistence.csv', index=False)
    if len(cr):
        summ = cr.groupby(['method', 'query']).reflects_correction.agg(['mean', 'sum', 'count']).reset_index()
        later = cr[cr['query'].isin(['Q2', 'Q3'])].groupby('method').reflects_correction.agg(['mean', 'sum', 'count']).rename(columns={'mean': 'retention_rate_Q2_Q3'})
        summ.to_csv(OUT / 'correction_persistence_summary.csv', index=False); print(summ.to_string(index=False)); print(later.to_string())
    print(df[df.metric.isin(['match_f1', 'status_weighted_f1', 'status_kappa', 'burden_accuracy'])].pivot(index='metric', columns='method', values='estimate').round(4).to_string())
    print(pdiff.round(4).to_string(index=False)); print(eff.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
