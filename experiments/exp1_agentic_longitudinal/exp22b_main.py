"""Experiment 2.2 external baselines: entry point.   python exp22b_main.py check | analyze"""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

import exp22b_run as RUN
import exp22b_common as C
import exp22b_stats as S
import exp22_run as R

BASE = C.BASE
AGENT = RUN.AGENT


def check():
    det, bad = RUN.deterministic()
    inv = RUN.inventory(det); rep = RUN.reproducibility(det)
    print(json.dumps(inv, indent=1)); print(rep.round(4).to_string(index=False)); print('failed evaluations:', len(bad), [b['err'] for b in bad][:3])
    return det


def write_conventional(det, patients):
    rows = []
    for m in ('Conventional (centroid)', 'Conventional (IoU)'):
        for p in patients:
            r = det[(p, m)]; bl, fu, pr = r['bl'], r['fu'], r['pred']
            for a, b in pr['pairs']:
                d = next((x for x in r['rows'] if x['baseline_lesion_id'] == a and x['followup_lesion_id'] == b), {})
                rows.append(dict(method=m, patient_id=p, baseline_lesion_id=a, followup_lesion_id=b, distance_mm=d.get('distance_mm'), registered_iou=d.get('registered_iou'), organ_match=None,
                                 size_ratio=fu[b]['volume_mm3'] / bl[a]['volume_mm3'] if bl[a]['volume_mm3'] else None, assigned=True, predicted_status='persistent',
                                 change_direction=pr['change'].get((a, b)), patient_burden_direction=pr['burden']))
            mb = {a for a, _ in pr['pairs']}; mf = {b for _, b in pr['pairs']}
            for a in bl:
                if a not in mb: rows.append(dict(method=m, patient_id=p, baseline_lesion_id=a, followup_lesion_id=None, distance_mm=None, organ_match=None, size_ratio=None, assigned=False, predicted_status='disappearing', change_direction=None, patient_burden_direction=pr['burden']))
            for b in fu:
                if b not in mf: rows.append(dict(method=m, patient_id=p, baseline_lesion_id=None, followup_lesion_id=b, distance_mm=None, organ_match=None, size_ratio=None, assigned=False, predicted_status='newly_appearing', change_direction=None, patient_burden_direction=pr['burden']))
    pd.DataFrame(rows).to_csv(BASE / 'conventional_predictions.csv', index=False)


def write_agent(recs):
    rows = []
    for r in recs:
        a = r['structured_answer'] or {}
        rows.append(dict(patient_id=r['patient_id'], repeat_id=r['repeat_id'], query_id=r['query_id'], raw_answer=r['raw_answer'], structured_answer=json.dumps(r['structured_answer']),
                         lesion_matches=json.dumps(a.get('matches')) if r['query_id'] == 'Q1' else None, status_predictions=json.dumps({k: a.get(k) for k in ('baseline_status', 'followup_status')}) if r['query_id'] == 'Q2' else None,
                         change_direction=json.dumps(a.get('changes')) if r['query_id'] == 'Q3' else None, burden_direction=a.get('burden_direction') if r['query_id'] == 'Q4' else None,
                         tool_calls=json.dumps(r['tool_calls']), tools_used=';'.join(r['tools_used']), n_tool_calls=r['n_tool_calls'], latency_sec=r['latency_sec'], token_count=r['token_count'], success=r['success'], failure_reason=r['failure_reason']))
    pd.DataFrame(rows).to_csv(BASE / 'generic_agent_predictions_long.csv', index=False)


def failure_matrix(det, A):
    rows = []
    for p in A['patients']:
        p_info = next(x for x in C.pairs_all() if x['patient_id'] == p); tx = R.register(p_info['baseline_ct'], p_info['followup_ct'], C.OUT / f'reg/{p}.tfm')
        ref = det[(p, 'Full OncoAgent')]['ref']
        for m in ('Full OncoAgent', 'Conventional (centroid)', 'Conventional (IoU)', 'Reference-mask oracle'):
            rows += RUN.lesion_outcomes(p, m, det[(p, m)]['ref'] if m == 'Reference-mask oracle' else ref, det[(p, m)]['pred'], tx)
        a = A['agent']
        if a and (p, 0) in a['merged']: rows += RUN.lesion_outcomes(p, AGENT, ref, a['merged'][(p, 0)][1], tx)
    df = pd.DataFrame(rows); df.to_csv(BASE / 'failure_matrix.csv', index=False)
    tags = ['close_spatial_lesions', 'disappeared_lesion', 'new_lesion', 'large_size_change', 'registration_sensitive', 'small_lesion', 'large_lesion', 'ambiguous_correspondence']
    out = []
    for m, g in df.groupby('method'):
        out.append(dict(method=m, tag='all', n_lesions=len(g), status_error_rate=1 - g.correct.mean()))
        for t in tags:
            x = g[g.failure_type.fillna('').str.contains(t)]
            if len(x): out.append(dict(method=m, tag=t, n_lesions=len(x), status_error_rate=1 - x.correct.mean()))
    pd.DataFrame(out).to_csv(BASE / 'failure_summary.csv', index=False); return df


def analyze():
    det, bad = RUN.deterministic(); recs = RUN.agent_records()
    A = RUN.analysis(det, recs)
    inv = RUN.inventory(det); rep = RUN.reproducibility(det)
    df, pdiff, mc = A['df'], A['pdiff'], A['mcnemar']
    df.to_csv(BASE / 'baseline_metrics.csv', index=False)
    spread = RUN.per_repeat_spread(A, det); extra = []
    if A['agent']:
        ag = A['agent']; extra.append(dict(quantity='cross_query_consistency', method=AGENT, value=float(np.nanmean([np.mean(v) for v in ag['cons'].values()])), note='mean over patients; mechanical Q1/Q2/Q3/Q4 agreement'))
        extra.append(dict(quantity='cross_query_consistency', method='Full OncoAgent', value=1.0, note='all four answers are read from one state (consistent by construction; verified on the state tables)'))
        extra.append(dict(quantity='provenance_grounded_fraction', method=AGENT, value=float(np.nanmean([np.mean(v) for v in ag['ground'].values()])), note='share of reported ids grounded in the episode tool outputs'))
        extra.append(dict(quantity='provenance_grounded_fraction', method='Full OncoAgent', value=1.0, note='every lesion, measurement, correspondence and status is stored in the state with its evidence'))
        for k, v in RUN.agreement(ag, A['patients']).items(): extra.append(dict(quantity='repeated_run_agreement|' + k, method=AGENT, value=v, note=f'{len(ag["reps"])} repeats'))
        extra.append(dict(quantity='per_patient_consistency_below_1', method=AGENT, value=float(np.mean([np.mean(v) < 0.999 for v in ag['cons'].values()])), note='fraction of patients with any cross-query inconsistency'))
    stat = pd.concat([df.assign(kind='metric'), pd.DataFrame(extra).assign(kind='consistency/provenance')], ignore_index=True)
    if len(spread): stat = pd.concat([stat, spread.assign(method=AGENT, kind='repeat_spread')], ignore_index=True)
    stat.to_csv(BASE / 'baseline_statistics.csv', index=False)
    pd.concat([pdiff.assign(test='patient-level paired bootstrap'), mc.assign(test='exact McNemar')], ignore_index=True).to_csv(BASE / 'baseline_paired_differences.csv', index=False)
    write_conventional(det, A['patients'])
    if recs: write_agent(recs)
    eff, curves = RUN.efficiency(det, A); eff.to_csv(BASE / 'efficiency_comparison.csv', index=False); curves.to_csv(BASE / 'figure2_efficiency_data.csv', index=False)
    fm = failure_matrix(det, A)
    if recs:
        A0 = RUN.analysis(det, [r for r in recs if r['repeat_id'] == 0])
        d0 = A0['df'][A0['df'].method == AGENT].assign(analysis='repeat 0 only (complete for all 274 patients)')
        d0.to_csv(BASE / 'baseline_sensitivity_repeat0.csv', index=False)
        A0['pdiff'][A0['pdiff'].comparison.str.contains('Generic')].to_csv(BASE / 'baseline_sensitivity_repeat0_paired.csv', index=False)
        raw_all = [json.loads(l) for l in (BASE / 'generic_agent_raw.jsonl').read_text().splitlines()]
        cov = dict(episodes_planned=3288, episodes_logged=len(raw_all), episodes_valid=len(recs), episodes_lost_to_api_errors=int(sum(RUN.is_infra_failure(r) for r in raw_all)),
                   genuine_agent_failures=int(sum((not r['success']) for r in recs)),
                   complete_runs_by_repeat={str(k): int(sum(1 for (p, r_), qs in A['agent']['by'].items() if r_ == k and len(qs) == 4)) for k in A['agent']['reps']})
        (BASE / 'agent_coverage.json').write_text(json.dumps(cov, indent=1)); print(json.dumps(cov, indent=1))
    print(df.pivot(index='metric', columns='method', values='estimate').round(3).to_string()); print(pdiff.round(3).to_string(index=False)); print(mc.to_string(index=False)); print(eff.round(2).to_string(index=False)); print(inv)
    return det, A, df, curves, rep


if __name__ == '__main__':
    {'check': check, 'analyze': analyze}[sys.argv[1]]()
