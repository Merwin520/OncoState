"""Experiment 2.2 external baselines: full comparison driver.

Stages (each writes its outputs into outputs/exp22_baselines/):
  inventory + reproducibility check of the regenerated 2.2 outputs against the frozen 2.2 numbers
  deterministic methods (Full, oracle, conventional x2) -> metrics, patient-level bootstrap, paired differences
  generic state-free agent (if generic_agent_raw.jsonl exists) -> same metrics, consistency, provenance, efficiency
  failure matrix, conventional_predictions.csv, generic_agent_predictions_long.csv, efficiency_comparison.csv
"""
import base64, hashlib, io, json, re, sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

import exp22b_common as C
import exp22b_methods as M
import exp22b_stats as S
import exp22_run as R

BASE = C.BASE
LOCKED = [p['patient_id'] for p in C.pairs_all() if p.get('cohort') == 'locked']
AGENT = 'Generic state-free agent'
ORDER = ['Full OncoAgent', AGENT, 'Conventional (centroid)', 'Conventional (IoU)', 'Reference-mask oracle']


def frozen_reference():
    """Frozen 2.2 point estimates (setting A, locked cohort) embedded in the final 2.2 figure script."""
    src = (C.ROOT / 'outputs/exp22_final/make_figure2.py').read_text()
    blob = re.search(r'_DATA = json.loads\(zlib.decompress\(base64.b64decode\(\s*((?:"[^"]*"\s*)+)\)\)', src).group(1)
    data = json.loads(zlib.decompress(base64.b64decode(''.join(re.findall(r'"([^"]*)"', blob)))).decode())
    st = pd.read_csv(io.StringIO(data['stats']))
    g = lambda m: float(st[(st.metric == m) & (st.contrast == 'A')].estimate.iloc[0])
    return dict(match_f1=g('match_f1'), match_precision=g('match_precision'), match_recall=g('match_recall'), status_weighted_f1=g('status_weighted_f1'),
                status_kappa=g('status_cohens_kappa'), change_accuracy_conditional=g('change_class_accuracy'), change_kappa_conditional=g('change_class_kappa'))


def deterministic():
    res = [r for r in M.run() if r['pid'] in set(LOCKED)]
    bad = [r for r in res if not r['ok']]
    return {(r['pid'], r['method']): r for r in res if r['ok']}, bad


def is_infra_failure(r):
    return (not r['success']) and (r.get('failure_reason') or '').startswith('RuntimeError: api')


def agent_records():
    """Valid episodes only. Episodes lost to API errors (no credits, rate limits) are 'not run', never counted as agent failures."""
    raw = BASE / 'generic_agent_raw.jsonl'
    if not raw.exists(): return []
    latest = {}
    for l in raw.read_text().splitlines():
        j = json.loads(l)
        if is_infra_failure(j): continue
        latest[(j['patient_id'], j['repeat_id'], j['query_id'])] = j
    return list(latest.values())


def agent_pred_parts(ans, qid):
    """Structured answer -> the partial prediction that this query is responsible for."""
    empty = dict(pairs=[], status_b={}, status_f={}, change={}, burden=None)
    if ans is None: return empty
    if qid == 'Q1': empty['pairs'] = [(m['baseline_lesion_id'], m['followup_lesion_id']) for m in ans['matches']]
    elif qid == 'Q2':
        empty['status_b'] = {int(k): v for k, v in ans['baseline_status'].items()}; empty['status_f'] = {int(k): v for k, v in ans['followup_status'].items()}
    elif qid == 'Q3':
        empty['pairs'] = [(m['baseline_lesion_id'], m['followup_lesion_id']) for m in ans['changes']]; empty['change'] = {(m['baseline_lesion_id'], m['followup_lesion_id']): m['change'] for m in ans['changes']}
    elif qid == 'Q4': empty['burden'] = ans['burden_direction']
    return empty


def agent_patient_ev(ref, recs):
    """Merge the four query outputs of one (patient, repeat) into one evaluation; a failed query contributes empty predictions (counted wrong)."""
    parts = {q: agent_pred_parts(recs[q]['structured_answer'] if q in recs and recs[q]['success'] else None, q) for q in ('Q1', 'Q2', 'Q3', 'Q4')}
    e1 = C.evaluate(ref, dict(parts['Q1'])); e2 = C.evaluate(ref, dict(parts['Q2'])); e3 = C.evaluate(ref, dict(parts['Q3'])); e4 = C.evaluate(ref, dict(parts['Q4']))
    ev = dict(tp=e1['tp'], pred_pairs=e1['pred_pairs'], gt_pers=e1['gt_pers'], conf=e2['conf'], chg_conf=e3['chg_conf'], chg_cond_n=e3['chg_cond_n'], chg_cond_ok=e3['chg_cond_ok'],
              chg_unc_n=e3['chg_unc_n'], chg_unc_ok=e3['chg_unc_ok'], burden_ok=e4['burden_ok'], burden_conf=e4['burden_conf'], burden_pred=e4['burden_pred'], burden_gt=e4['burden_gt'])
    merged_pred = dict(pairs=parts['Q1']['pairs'], status_b=parts['Q2']['status_b'], status_f=parts['Q2']['status_f'], change=parts['Q3']['change'], burden=parts['Q4']['burden'])
    return ev, merged_pred


def consistency(recs_by_q, pred_parts):
    """Cross-query consistency within one (patient, repeat): Q1 matches vs Q2 statuses vs Q3 pairs vs Q4 totals (mechanical checks)."""
    scores = []
    q1, q2, q3, q4 = (recs_by_q.get(q) for q in ('Q1', 'Q2', 'Q3', 'Q4'))
    if q1 and q1['success'] and q2 and q2['success']:
        mb = {m['baseline_lesion_id'] for m in q1['structured_answer']['matches']}; mf = {m['followup_lesion_id'] for m in q1['structured_answer']['matches']}
        sb = {int(k): v for k, v in q2['structured_answer']['baseline_status'].items()}; sf = {int(k): v for k, v in q2['structured_answer']['followup_status'].items()}
        agree = [(sb.get(i) == 'persistent') == (i in mb) for i in set(sb) | mb] + [(sf.get(i) == 'persistent') == (i in mf) for i in set(sf) | mf]
        scores.append(np.mean(agree) if agree else 1.0)
    if q1 and q1['success'] and q3 and q3['success']:
        a = {(m['baseline_lesion_id'], m['followup_lesion_id']) for m in q1['structured_answer']['matches']}; b = {(m['baseline_lesion_id'], m['followup_lesion_id']) for m in q3['structured_answer']['changes']}
        scores.append(len(a & b) / len(a | b) if (a | b) else 1.0)
    if q4 and q4['success']:
        s = q4['structured_answer']; d = 'increase' if s['followup_total_volume_mm3'] > s['baseline_total_volume_mm3'] else 'decrease' if s['followup_total_volume_mm3'] < s['baseline_total_volume_mm3'] else 'no_change'
        scores.append(float(d == s['burden_direction']))
    return float(np.mean(scores)) if scores else np.nan


def grounded_fraction(recs_by_q):
    """Share of reported lesion ids / pairs that appear in that episode's own tool outputs (mechanical provenance check)."""
    vals = []
    for q, r in recs_by_q.items():
        if not r['success']: vals.append(0.0); continue
        a = r['structured_answer']; sb, sf = set(r['seen_baseline_ids']), set(r['seen_followup_ids'])
        used_calls = {c['name'] for c in r['tool_calls'] if c['ok']}
        if q in ('Q1', 'Q3'):
            key = 'matches' if q == 'Q1' else 'changes'; items = a[key]
            vals.append(np.mean([(m['baseline_lesion_id'] in sb or 'match_lesions' in used_calls) and (m['followup_lesion_id'] in sf or 'match_lesions' in used_calls) for m in items]) if items else 1.0)
        elif q == 'Q2':
            vals.append(float(all(int(k) in sb or 'match_lesions' in used_calls for k in a['baseline_status']) and all(int(k) in sf or 'match_lesions' in used_calls for k in a['followup_status'])))
        else:
            vals.append(float('list_lesions' in used_calls and 'calculate' in used_calls or 'list_lesions' in used_calls))
    return float(np.mean(vals)) if vals else np.nan


def lesion_outcomes(pid, method, ref, pred, tx):
    """One row per reference lesion: reference status, predicted status, correctness and failure tags."""
    rows = []; cent_b, cent_f = ref['gt_cent_b'], ref['gt_cent_f']
    def nearest(cent, g):
        others = [np.linalg.norm(np.array(cent[g]) - np.array(c)) for h, c in cent.items() if h != g]
        return min(others) if others else np.inf
    reg_d = {}
    for g in set(cent_b) & set(cent_f):
        reg_d[g] = float(np.linalg.norm(np.array(tx.TransformPoint(tuple(cent_b[g]))) - np.array(cent_f[g])))
    reg_quality = float(np.median(list(reg_d.values()))) if reg_d else np.nan
    sb, sf = pred.get('status_b') or {}, pred.get('status_f') or {}
    for g, cls in ref['gclass'].items():
        pb, pf = ref['inv_b'].get(g), ref['inv_f'].get(g)
        if cls == 'persistent': st = sb.get(pb) if pb is not None and pb in sb else (sf.get(pf) if pf is not None and pf in sf else None)
        elif cls == 'disappearing': st = sb.get(pb) if pb is not None else None
        else: st = sf.get(pf) if pf is not None else None
        st = st if st in C.STATUS else 'undetected'
        vb, vf = ref['gt_vol_b'].get(g), ref['gt_vol_f'].get(g); size = max(v for v in (vb, vf) if v is not None) / 1000
        tags = []
        if min(nearest(cent_b, g) if g in cent_b else np.inf, nearest(cent_f, g) if g in cent_f else np.inf) < 20: tags.append('close_spatial_lesions')
        if cls == 'disappearing': tags.append('disappeared_lesion')
        if cls == 'newly_appearing': tags.append('new_lesion')
        if cls == 'persistent' and vb and vf and abs((vf - vb) / vb * 100) > 50: tags.append('large_size_change')
        if g in reg_d and reg_d[g] > 30: tags.append('registration_sensitive')
        if size < 0.5: tags.append('small_lesion')
        if size > 10: tags.append('large_lesion')
        if cls == 'persistent' and g in reg_d:
            n_close = sum(np.linalg.norm(np.array(tx.TransformPoint(tuple(cent_b[g]))) - np.array(c)) <= 30 for c in cent_f.values())
            if n_close >= 2: tags.append('ambiguous_correspondence')
        rows.append(dict(patient_id=pid, lesion_id=g, method=method, reference=cls, prediction=st, correct=st == cls, failure_type=';'.join(tags), distance=reg_d.get(g, np.nan), size=size, organ='not available',
                         baseline_volume=vb, followup_volume=vf, registration_quality=reg_quality))
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------
def inventory(det):
    full = {p: det[(p, 'Full OncoAgent')] for p in LOCKED if (p, 'Full OncoAgent') in det}
    n_ref = sum(len(r['ref']['gclass']) for r in full.values()); n_pers = sum(sum(c == 'persistent' for c in r['ref']['gclass'].values()) for r in full.values())
    inv = dict(n_patients=len(full), n_studies=2 * len(full), n_reference_lesions=n_ref, n_longitudinal_pairs=n_pers,
               n_available_masks=int(len(list((C.OUT / 'cut_A').glob('*.nii.gz')))), n_registered_lesions=int(sum(len(r['bl']) for r in full.values())),
               n_reference_matches=n_pers, n_status_labels=n_ref, n_patient_burden_labels=len(full),
               patient_ids_sha256=hashlib.sha256(','.join(sorted(full)).encode()).hexdigest(),
               source={'lesion_masks': str(C.OUT / 'cut_A'), 'registration_transforms': str(C.OUT / 'reg'), 'states': [str(C.OUT / 'state_A'), str(C.OUT / 'state_S0')],
                       'regenerated_with': 'frozen Experiment 2.2 scripts (exp22_prep/run/cut_worker); original intermediates had been deleted'})
    (BASE / 'exp22_data_inventory.json').write_text(json.dumps(inv, indent=1)); return inv


def reproducibility(det):
    """Compare the regenerated Full/A outputs with the frozen 2.2 numbers (same definitions, locked cohort)."""
    per = {p: S.vec(det[(p, 'Full OncoAgent')]['ev']) for p in LOCKED if (p, 'Full OncoAgent') in det}
    tot = np.sum(list(per.values()), 0); m = {k: float(v) for k, v in S.metrics_from_sums(tot, 1.0).items()}
    fr = frozen_reference(); rows = [dict(metric=k, frozen=fr[k], regenerated=m[k], abs_diff=abs(fr[k] - m[k])) for k in fr]
    df = pd.DataFrame(rows); df.to_csv(BASE / 'reproducibility_check.csv', index=False); return df


def analysis(det, recs):
    patients = [p for p in LOCKED if all((p, m) in det for m in ('Full OncoAgent', 'Reference-mask oracle', 'Conventional (centroid)', 'Conventional (IoU)'))]
    per, succ, evs = {}, {}, {}
    for m in ('Full OncoAgent', 'Reference-mask oracle', 'Conventional (centroid)', 'Conventional (IoU)'):
        per[m] = {p: S.vec(det[(p, m)]['ev']) for p in patients}; succ[m] = {p: 1.0 for p in patients}; evs[m] = {p: det[(p, m)]['ev'] for p in patients}
    agent = {}
    if recs:
        by = {}
        for r in recs: by.setdefault((r['patient_id'], r['repeat_id']), {})[r['query_id']] = r
        reps = sorted({k[1] for k in by})
        vecs = {p: [] for p in patients}; sc = {p: [] for p in patients}; cons = {p: [] for p in patients}; ground = {p: [] for p in patients}; b_ok = {p: [] for p in patients}
        per_rep_metrics = {r: [] for r in reps}; merged = {}
        for (p, r), qs in by.items():
            if p not in vecs or len(qs) < 4: continue
            ev, pred = agent_patient_ev(det[(p, 'Full OncoAgent')]['ref'], qs); merged[(p, r)] = (ev, pred, qs)
            vecs[p].append(S.vec(ev)); sc[p].append(float(all(qs[q]['success'] for q in qs))); cons[p].append(consistency(qs, pred)); ground[p].append(grounded_fraction(qs)); b_ok[p].append(ev['burden_ok'])
        patients_agent = [p for p in patients if vecs[p]]
        if len(patients_agent) == len(patients):
            per[AGENT] = {p: np.mean(vecs[p], 0) for p in patients}; succ[AGENT] = {p: float(np.mean(sc[p])) for p in patients}
            agent = dict(by=by, reps=reps, cons=cons, ground=ground, merged=merged, burden_ok={p: float(np.mean([x for x in b_ok[p] if x is not None])) if any(x is not None for x in b_ok[p]) else 0.0 for p in patients})
    df, point, boot, idx, Mx, Sc = S.analyse(per, succ, patients)
    df['method'] = pd.Categorical(df.method, [m for m in ORDER if m in per], ordered=True); df = df.sort_values(['method', 'metric'])
    pdiff = S.paired(point, boot)
    # exact McNemar on patient-level burden correctness and full task success, Full vs each deployable baseline
    mc = []
    burden = {m: {p: float(evs[m][p]['burden_ok']) for p in patients} for m in evs}
    if AGENT in per: burden[AGENT] = agent['burden_ok']
    for m in burden:
        if m == 'Full OncoAgent': continue
        mc.append(S.mcnemar_rows(patients, burden['Full OncoAgent'], burden[m], f'Full OncoAgent vs {m}', 'patient burden direction correct'))
    mc = pd.DataFrame(mc); mc['bh_q'] = S.bh(mc.p_value.values) if len(mc) else []
    return dict(patients=patients, per=per, succ=succ, evs=evs, df=df, point=point, boot=boot, idx=idx, pdiff=pdiff, mcnemar=mc, agent=agent)


def per_repeat_spread(A, det):
    """Mean +- SD of every metric across the generic agent's repeats. Each repeat is pooled over the SAME patients (those with every repeat complete),
    so the spread reflects run-to-run variation, not different patient subsets."""
    a = A['agent']; rows = []
    if not a: return pd.DataFrame(rows)
    common = [p for p in A['patients'] if all((p, r) in a['merged'] for r in a['reps'])]
    if not common: return pd.DataFrame(rows)
    for r in a['reps']:
        S_ = np.sum([S.vec(a['merged'][(p, r)][0]) for p in common], 0); succ = np.mean([float(all(a['merged'][(p, r)][2][q]['success'] for q in a['merged'][(p, r)][2])) for p in common])
        m = S.metrics_from_sums(S_, succ)
        for k in S.METRICS: rows.append(dict(repeat_id=r, metric=k, value=float(m[k])))
    d = pd.DataFrame(rows); out = d.groupby('metric').value.agg(['mean', 'std']).reset_index().rename(columns={'mean': 'mean_over_repeats', 'std': 'sd_over_repeats'})
    out['n_patients_all_repeats_complete'] = len(common); return out


def agreement(a, patients):
    """Repeated-run agreement of the generic agent (mean over patients and repeat pairs)."""
    out = {'Q1_pair_jaccard': [], 'Q2_status_agreement': [], 'Q3_change_agreement': [], 'Q4_direction_agreement': []}
    for p in patients:
        rs = [r for r in a['reps'] if (p, r) in a['merged']]
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                A_, B_ = a['merged'][(p, rs[i])][1], a['merged'][(p, rs[j])][1]
                sa, sb = set(A_['pairs']), set(B_['pairs']); out['Q1_pair_jaccard'].append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)
                keys = set(A_['status_b']) | set(B_['status_b']); out['Q2_status_agreement'].append(np.mean([A_['status_b'].get(k) == B_['status_b'].get(k) for k in keys]) if keys else 1.0)
                ck = set(A_['change']) | set(B_['change']); out['Q3_change_agreement'].append(np.mean([A_['change'].get(k) == B_['change'].get(k) for k in ck]) if ck else 1.0)
                out['Q4_direction_agreement'].append(float(A_['burden'] == B_['burden']))
    return {k: float(np.mean(v)) if v else np.nan for k, v in out.items()}


def efficiency(det, A):
    """State reconstruction, tool calls, tokens and latency per patient over the four sequential queries; cumulative curve per query index."""
    rows, curves = [], []
    ptime = {}
    for p in A['patients']:
        try:
            a_rt = json.loads((C.OUT / f'state_A/{p}.json').read_text())['runtime_seconds']; s_rt = json.loads((C.OUT / f'state_S0/{p}.json').read_text())['runtime_seconds']
            ptime[p] = dict(match=a_rt, reg=max(s_rt - a_rt, 0.0))
        except Exception: pass
    mean_match = np.mean([v['match'] for v in ptime.values()]); mean_reg = np.mean([v['reg'] for v in ptime.values()])
    n = len(A['patients'])
    for m in ('Full OncoAgent', 'Conventional (centroid)', 'Conventional (IoU)', 'Reference-mask oracle'):
        rows.append(dict(method=m, state_constructions_per_patient=1, registration_invocations=1, matching_invocations=1, measurement_invocations=2, total_tool_calls=4, llm_calls=0, tokens=0,
                         llm_latency_sec=0.0, est_uncached_compute_sec=mean_reg + mean_match, cross_query_reuse='state read by all 4 queries' if m == 'Full OncoAgent' else 'single pass; no queryable state'))
    for q in range(1, 5):
        curves.append(dict(method='Full OncoAgent', query_index=q, cumulative_tool_calls=4.0, cumulative_state_constructions=1.0, cumulative_est_compute_sec=mean_reg + mean_match, cumulative_llm_latency_sec=0.0, cumulative_tokens=0.0))
    a = A['agent']
    if a:
        tot = {k: [] for k in ('calls', 'reg', 'match', 'meas', 'llm', 'tokens', 'lat', 'comp')}; cum = {q: {k: [] for k in ('calls', 'recon', 'comp', 'lat', 'tokens')} for q in range(1, 5)}
        for (p, r), qs in a['by'].items():
            if p not in A['patients'] or len(qs) < 4: continue
            run = dict(calls=0, recon=0, comp=0.0, lat=0.0, tokens=0)
            for qi, q in enumerate(('Q1', 'Q2', 'Q3', 'Q4'), 1):
                tc = qs[q]['tool_calls']; run['calls'] += len(tc); run['recon'] += int(any(c['name'] == 'match_lesions' for c in tc))
                run['comp'] += sum(mean_reg if c['name'] == 'register_studies' else mean_match if c['name'] == 'match_lesions' else 0.0 for c in tc) + sum(c['compute_seconds_if_uncached'] for c in tc if c['name'] in ('list_lesions', 'get_study_info'))
                run['lat'] += qs[q]['latency_sec']; run['tokens'] += qs[q]['token_count']
                for k in run: cum[qi][k].append(run[k])
            allc = [c for q in qs for c in qs[q]['tool_calls']]
            tot['calls'].append(len(allc)); tot['reg'].append(sum(c['name'] == 'register_studies' for c in allc)); tot['match'].append(sum(c['name'] == 'match_lesions' for c in allc))
            tot['meas'].append(sum(c['name'] in ('list_lesions', 'get_study_info') for c in allc)); tot['llm'].append(sum(qs[q]['n_turns'] for q in qs)); tot['tokens'].append(sum(qs[q]['token_count'] for q in qs)); tot['lat'].append(sum(qs[q]['latency_sec'] for q in qs)); tot['comp'].append(run['comp'])
        rows.append(dict(method=AGENT, state_constructions_per_patient=float(np.mean(cum[4]['recon'])), registration_invocations=float(np.mean(tot['reg'])), matching_invocations=float(np.mean(tot['match'])), measurement_invocations=float(np.mean(tot['meas'])),
                         total_tool_calls=float(np.mean(tot['calls'])), llm_calls=float(np.mean(tot['llm'])), tokens=float(np.mean(tot['tokens'])), llm_latency_sec=float(np.mean(tot['lat'])), est_uncached_compute_sec=float(np.mean(tot['comp'])),
                         cross_query_reuse='none (each query is a fresh episode)'))
        for qi in range(1, 5):
            curves.append(dict(method=AGENT, query_index=qi, cumulative_tool_calls=float(np.mean(cum[qi]['calls'])), cumulative_state_constructions=float(np.mean(cum[qi]['recon'])), cumulative_est_compute_sec=float(np.mean(cum[qi]['comp'])),
                               cumulative_llm_latency_sec=float(np.mean(cum[qi]['lat'])), cumulative_tokens=float(np.mean(cum[qi]['tokens'])), tool_calls_sd_over_patients=float(np.std(cum[qi]['calls'])), n_runs=len(cum[qi]['calls'])))
    return pd.DataFrame(rows), pd.DataFrame(curves)
