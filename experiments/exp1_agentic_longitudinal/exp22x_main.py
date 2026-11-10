"""Experiment 2.2 external comparison: assemble every system, evaluate, and write all tables.   python exp22x_main.py"""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

import exp22b_common as C
import exp22b_methods as M
import exp22b_run as RUN
import exp22x_eval as X

BASE = X.BASE
LOCKED = RUN.LOCKED
VLM_MODELS = {'gpt-5.5': 'Direct VLM A (gpt-5.5)', 'gpt-5.2': 'Direct VLM B (gpt-5.2)', 'gpt-5.4-mini': 'Direct VLM C (gpt-5.4-mini)'}
MEDVLM_MODELS = {'llava_med': 'Direct medical VLM (LLaVA-Med v1.5 7B)', 'medgemma': 'Direct medical VLM (MedGemma 4B-it)'}
AGENT2 = 'Generic tool-using agent (gpt-5.2)'
DET = {'Full OncoAgent': 'Full OncoAgent', 'Conventional (centroid)': 'Conventional pipeline (registered centroid + Hungarian)', 'Conventional (IoU)': 'Conventional (registered IoU) [supplementary]',
       'Reference-mask oracle': 'Reference-mask oracle (upper control)'}
ORDER = [DET['Full OncoAgent'], *VLM_MODELS.values(), *MEDVLM_MODELS.values(), AGENT2, DET['Conventional (centroid)'], DET['Conventional (IoU)'], DET['Reference-mask oracle']]


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def load_all():
    det, _ = RUN.deterministic()
    refs = {p: det[(p, 'Full OncoAgent')]['ref'] for p in LOCKED if (p, 'Full OncoAgent') in det}
    keys = {p: json.loads((BASE / 'packets' / p / 'private_key.json').read_text()) for p in LOCKED if (BASE / 'packets' / p / 'private_key.json').exists()}
    return det, refs, keys


def run_vectors(det, refs, keys):
    """{(system, model, repeat): {pid: (vec, valid, api_success)}}"""
    out = {}
    for m, name in DET.items():
        out[(name, m, 0)] = {p: (X.evaluate(det[(p, m)]['ref'] if m == 'Reference-mask oracle' else refs[p], X.det_pred(det[(p, m)])), True, True) for p in LOCKED if (p, m) in det}
    for r in read_jsonl(BASE / 'direct_vlm_raw.jsonl'):
        if not r.get('api_success') or r['patient_id'] not in refs: continue
        pr = X.packet_pred(r['structured_answer'], keys[r['patient_id']]); out.setdefault((VLM_MODELS[r['model_id']], r['model_id'], r['repeat_id']), {})[r['patient_id']] = (X.evaluate(refs[r['patient_id']], pr), bool(r.get('valid')), True)
    for r in read_jsonl(BASE / 'generic_agent_v2_raw.jsonl'):
        if not r.get('api_success') or r['patient_id'] not in refs: continue
        pr = X.packet_pred(r['structured_answer'], keys[r['patient_id']]); out.setdefault((AGENT2, 'gpt-5.2', r['repeat_id']), {})[r['patient_id']] = (X.evaluate(refs[r['patient_id']], pr), bool(r.get('valid')), True)
    for r in read_jsonl(BASE / 'medical_vlm_raw.jsonl'):
        if not r.get('run_success') or r['model_key'] not in MEDVLM_MODELS or r['patient_id'] not in refs: continue
        pr = X.packet_pred(r['structured_answer'], keys[r['patient_id']]); out.setdefault((MEDVLM_MODELS[r['model_key']], r['model_key'], r['repeat_id']), {})[r['patient_id']] = (X.evaluate(refs[r['patient_id']], pr), bool(r.get('valid')), True)
    return out


def matrices(runs, repeat):
    mats, avail, valid = {}, {}, {}
    for (name, mid, rep), d in runs.items():
        if rep != repeat: continue
        mats[name] = np.stack([d[p][0] if p in d else np.zeros(X.D) for p in LOCKED]); avail[name] = np.array([p in d for p in LOCKED]); valid[name] = np.array([d[p][1] if p in d else False for p in LOCKED])
    return mats, avail, valid


def main():
    det, refs, keys = load_all(); LOCKED_OK = [p for p in LOCKED if p in refs and p in keys]
    runs = run_vectors(det, refs, keys)
    mats, avail, valid = matrices(runs, 0); names = [n for n in ORDER if n in mats]
    point, boot, counts = X.bootstrap(mats, avail, LOCKED)
    planned = {n: len(LOCKED) for n in names}; rows = []
    for n in names:
        ne = int(avail[n].sum()); ok_valid = int((avail[n] & valid[n]).sum())
        for k in X.METRICS: rows.append(dict(method=n, metric=k, estimate=point[n][k], ci95_lo=np.nanpercentile(boot[n][k], 2.5), ci95_hi=np.nanpercentile(boot[n][k], 97.5), n_evaluated=ne, n_planned=planned[n]))
        rows.append(dict(method=n, metric='api_coverage', estimate=ne / planned[n], n_evaluated=ne, n_planned=planned[n]))
        rows.append(dict(method=n, metric='conditional_task_success', estimate=ok_valid / ne if ne else np.nan, n_evaluated=ne, n_planned=planned[n]))
    met = pd.DataFrame(rows); met.to_csv(BASE / 'external_metrics.csv', index=False)
    # paired differences (repeat 0, common subset), BH across all McNemar tests
    pdiff = []
    for n in names:
        if n != DET['Full OncoAgent']: pdiff += X.paired_common(mats, avail, DET['Full OncoAgent'], n, counts)
    pdiff = pd.DataFrame(pdiff); mc = []
    for n in names:
        if n == DET['Full OncoAgent']: continue
        com = avail[DET['Full OncoAgent']] & avail[n]
        for label, col in (('burden direction correct', None),):
            fa = np.array([mats[DET['Full OncoAgent']][i, 34:46].reshape(3, 4).trace() for i in range(len(LOCKED))]) > 0; fb = np.array([mats[n][i, 34:46].reshape(3, 4).trace() for i in range(len(LOCKED))]) > 0
            n10, n01 = int((fa & ~fb & com).sum()), int((~fa & fb & com).sum()); mc.append(dict(comparison=f'Full OncoAgent vs {n}', endpoint=label, only_full_correct=n10, only_other_correct=n01, p_value=binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0, n_common=int(com.sum())))
    mc = pd.DataFrame(mc)
    if len(mc):
        ps = mc.p_value.values; o = np.argsort(ps); q = np.empty(len(ps)); prev = 1.0
        for rank, i in zip(range(len(ps), 0, -1), o[::-1]): prev = min(prev, ps[i] * len(ps) / rank); q[i] = prev
        mc['bh_q'] = q
    pd.concat([pdiff.assign(test='paired patient-level bootstrap (common subset, repeat 0)'), mc.assign(test='exact McNemar')], ignore_index=True).to_csv(BASE / 'external_paired_differences.csv', index=False)
    # common-subset sensitivity: patients evaluated by EVERY system
    common = np.all([avail[n] for n in names], 0); sens = []
    for n in names:
        m_ = metrics_common(mats[n], common)
        for k in X.METRICS: sens.append(dict(method=n, metric=k, estimate=m_[k], n_common=int(common.sum())))
    pd.DataFrame(sens).to_csv(BASE / 'external_common_subset_sensitivity.csv', index=False)
    # repeat spread (mean +- SD over repeats, each repeat on the patients it evaluated)
    rep_rows = []
    for (name, mid, rep), d in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        if name not in names or name in (DET['Full OncoAgent'], DET['Conventional (centroid)'], DET['Conventional (IoU)'], DET['Reference-mask oracle']): continue
        S_ = np.sum([v[0] for v in d.values()], 0); mm = X.metrics(S_)
        for k in ('match_f1', 'change_accuracy', 'burden_accuracy'): rep_rows.append(dict(method=name, repeat_id=rep, metric=k, value=float(mm[k]), n_patients=len(d), conditional_success=float(np.mean([v[1] for v in d.values()]))))
    rep_df = pd.DataFrame(rep_rows); rep_df.to_csv(BASE / 'external_repeat_metrics.csv', index=False)
    consistency(runs, keys, refs)
    write_predictions(det, refs, keys)
    failure_matrix(det, refs, keys)
    main_table(met, rep_df, names)
    print(met.pivot(index='metric', columns='method', values='estimate')[names].round(3).to_string())
    print(met[met.metric == 'api_coverage'][['method', 'n_evaluated', 'n_planned']].to_string(index=False))
    return det, refs, keys, runs


def metrics_common(mat, common): return {k: float(v) for k, v in X.metrics((mat * common[:, None]).sum(0)).items()}


def consistency(runs, keys, refs):
    """Repeat consistency of the stochastic systems (needs >= 2 repeats): correspondence, change, burden, complete-patient, pairwise, Fleiss kappa."""
    rows = []; raw = {'vlm': read_jsonl(BASE / 'direct_vlm_raw.jsonl'), 'agent': read_jsonl(BASE / 'generic_agent_v2_raw.jsonl')}
    by = {}
    for r in raw['vlm']:
        if r.get('api_success'): by.setdefault(VLM_MODELS[r['model_id']], {}).setdefault(r['patient_id'], {})[r['repeat_id']] = r['structured_answer']
    for r in raw['agent']:
        if r.get('api_success'): by.setdefault(AGENT2, {}).setdefault(r['patient_id'], {})[r['repeat_id']] = r['structured_answer']
    for name, pats in by.items():
        match_c, chg_c, bur_c, full_c, pair_agree, bur_items, match_items = [], [], [], [], [], [], []
        for p, reps in pats.items():
            if len(reps) < 3 or any(a is None for a in reps.values()): continue
            ms = {r: {m['baseline_id']: (m['followup_id'], m['change']) for m in a['matches']} for r, a in reps.items()}; bids = sorted(set().union(*[set(x) for x in ms.values()]))
            same_match = [len({ms[r].get(b, ('none', ''))[0] for r in ms}) == 1 for b in bids]; same_change = [len({ms[r].get(b, ('none', ''))[1] for r in ms}) == 1 for b in bids]
            same_bur = len({a['burden_direction'] for a in reps.values()}) == 1
            match_c.append(np.mean(same_match) if same_match else 1.0); chg_c.append(np.mean(same_change) if same_change else 1.0); bur_c.append(float(same_bur)); full_c.append(float(all(same_match) and all(same_change) and same_bur))
            for r1, r2 in combinations(sorted(ms), 2): pair_agree.append(np.mean([ms[r1].get(b, ('none',))[0] == ms[r2].get(b, ('none',))[0] for b in bids]) if bids else 1.0)
            bur_items.append([reps[r]['burden_direction'] for r in sorted(reps)]); match_items += [[ms[r].get(b, ('none',))[0] for r in sorted(ms)] for b in bids]
        def fleiss(items):
            if not items: return np.nan
            cats = sorted({c for it in items for c in it}); n = len(items[0]); N = len(items)
            Pm = np.array([[it.count(c) for c in cats] for it in items], float); Pi = ((Pm ** 2).sum(1) - n) / (n * (n - 1)); pj = Pm.sum(0) / (N * n); pe = (pj ** 2).sum()
            return float((Pi.mean() - pe) / (1 - pe)) if pe < 1 else np.nan
        if full_c: rows.append(dict(method=name, n_patients_with_3_valid_repeats=len(full_c), correspondence_consistency=float(np.mean(match_c)), change_consistency=float(np.mean(chg_c)), burden_consistency=float(np.mean(bur_c)),
                                    fraction_patients_fully_consistent=float(np.mean(full_c)), pairwise_match_agreement=float(np.mean(pair_agree)), fleiss_kappa_burden=fleiss(bur_items), fleiss_kappa_match=fleiss(match_items)))
    pd.DataFrame(rows).to_csv(BASE / 'external_consistency.csv', index=False)


def write_predictions(det, refs, keys):
    rows = []
    for r in read_jsonl(BASE / 'direct_vlm_raw.jsonl'):
        if not r.get('api_success'): continue
        a = r['structured_answer']
        for m in (a['matches'] if a else [None]):
            rows.append(dict(model_id=r['model_id'], api_model=r.get('api_model'), patient_id=r['patient_id'], repeat_id=r['repeat_id'], valid=r.get('valid'), baseline_id=m and m['baseline_id'], followup_id=m and m['followup_id'], change=m and m['change'],
                             burden_direction=a and a['burden_direction'], latency_sec=r.get('latency_sec'), token_count=r.get('token_count'), schema_repair_used=r.get('schema_repair_used')))
    pd.DataFrame(rows).to_csv(BASE / 'direct_vlm_predictions.csv', index=False)
    rows = []
    for r in read_jsonl(BASE / 'generic_agent_v2_raw.jsonl'):
        if not r.get('api_success'): continue
        a = r['structured_answer']
        for m in (a['matches'] if a else [None]):
            rows.append(dict(patient_id=r['patient_id'], repeat_id=r['repeat_id'], valid=r.get('valid'), baseline_id=m and m['baseline_id'], followup_id=m and m['followup_id'], change=m and m['change'], burden_direction=a and a['burden_direction'],
                             tools_used=';'.join(r['tools_used']), n_tool_calls=r['n_tool_calls'], n_llm_calls=r['n_turns'], latency_sec=r['latency_sec'], token_count=r['token_count'], status=r['status']))
    pd.DataFrame(rows).to_csv(BASE / 'generic_agent_predictions.csv', index=False)
    rows = []
    for r in read_jsonl(BASE / 'medical_vlm_raw.jsonl'):
        if not r.get('run_success') or r['model_key'] not in MEDVLM_MODELS: continue
        a = r['structured_answer']
        for m in (a['matches'] if a else [None]):
            rows.append(dict(model_key=r['model_key'], model_id=r.get('model_id'), patient_id=r['patient_id'], repeat_id=r['repeat_id'], valid=r.get('valid'), baseline_id=m and m['baseline_id'], followup_id=m and m['followup_id'], change=m and m['change'],
                             burden_direction=a and a['burden_direction'], latency_sec=r.get('latency_sec'), n_images=r.get('n_images'), packet_truncated=r.get('packet_truncated'), schema_repair_used=r.get('schema_repair_used')))
    pd.DataFrame(rows).to_csv(BASE / 'medical_vlm_predictions.csv', index=False)
    for m, fname in (('Full OncoAgent', 'full_predictions.csv'), ('Conventional (centroid)', 'conventional_predictions.csv'), ('Reference-mask oracle', 'oracle_predictions.csv')):
        rows = []
        for p in LOCKED:
            if (p, m) not in det: continue
            r = det[(p, m)]; pr = X.det_pred(r)
            for (gb, gf) in pr['pairs']: rows.append(dict(patient_id=p, method=m, reference_baseline_lesion=gb, reference_followup_lesion=gf, match_correct=gb is not None and gb == gf, change=pr['change'].get((gb, gf)), burden_direction=pr['burden']))
            if not pr['pairs']: rows.append(dict(patient_id=p, method=m, burden_direction=pr['burden']))
        pd.DataFrame(rows).to_csv(BASE / fname, index=False)


def failure_matrix(det, refs, keys):
    import exp22_run as R
    rows = []; vlm = {}
    for r in read_jsonl(BASE / 'direct_vlm_raw.jsonl'):
        if r.get('api_success') and r['repeat_id'] == 0: vlm[(r['model_id'], r['patient_id'])] = r
    ag = {r['patient_id']: r for r in read_jsonl(BASE / 'generic_agent_v2_raw.jsonl') if r.get('api_success') and r['repeat_id'] == 0}
    mv = {}
    for r in read_jsonl(BASE / 'medical_vlm_raw.jsonl'):
        if r.get('run_success') and r['model_key'] in MEDVLM_MODELS: mv[(r['model_key'], r['patient_id'])] = r
    for p in LOCKED:
        if p not in refs or p not in keys: continue
        ref = refs[p]; pinfo = next(x for x in C.pairs_all() if x['patient_id'] == p); tx = R.register(pinfo['baseline_ct'], pinfo['followup_ct'], C.OUT / f'reg/{p}.tfm')
        reg_d = {g: float(np.linalg.norm(np.array(tx.TransformPoint(tuple(ref['gt_cent_b'][g]))) - np.array(ref['gt_cent_f'][g]))) for g in set(ref['gt_cent_b']) & set(ref['gt_cent_f'])}
        regq = float(np.median(list(reg_d.values()))) if reg_d else np.nan
        systems = {DET[m]: (m, X.det_pred(det[(p, m)])) for m in DET if (p, m) in det}
        for mid, name in VLM_MODELS.items():
            if (mid, p) in vlm: systems[name] = (mid, X.packet_pred(vlm[(mid, p)]['structured_answer'], keys[p]))
        if p in ag: systems[AGENT2] = ('gpt-5.2', X.packet_pred(ag[p]['structured_answer'], keys[p]))
        for mk, name in MEDVLM_MODELS.items():
            if (mk, p) in mv: systems[name] = (mk, X.packet_pred(mv[(mk, p)]['structured_answer'], keys[p]))
        for name, (mid, pr) in systems.items():
            correct = {gb for gb, gf in pr['pairs'] if gb is not None and gb == gf}; mapped_b = {gb: gf for gb, gf in pr['pairs'] if gb is not None}
            for g, cls in ref['gclass'].items():
                vb, vf = ref['gt_vol_b'].get(g), ref['gt_vol_f'].get(g); size = max(v for v in (vb, vf) if v is not None) / 1000
                cb, cf = ref['gt_cent_b'], ref['gt_cent_f']
                near = min([np.linalg.norm(np.array(cb[g]) - np.array(c)) for h, c in cb.items() if h != g] + [np.inf]) if g in cb else np.inf
                tags = [t for t, cond in (('small_lesion', size < 0.5), ('large_lesion', size > 10), ('close_neighbouring_lesion', near < 20), ('disappeared_lesion', cls == 'disappearing'), ('apparent_new_lesion', cls == 'newly_appearing'),
                                           ('large_size_change', cls == 'persistent' and vb and vf and abs((vf - vb) / vb * 100) > 50), ('registration_sensitive', g in reg_d and reg_d[g] > 30), ('high_lesion_count', len(ref['gclass']) >= 20)) if cond]
                if cls == 'persistent': mc = g in correct
                elif cls == 'disappearing': mc = g not in mapped_b or mapped_b[g] is None
                else: mc = not any(gf == g for _, gf in pr['pairs'])
                rows.append(dict(patient_id=p, lesion_id=g, method=name, model_id=mid, repeat_id=0, reference_match=(cls == 'persistent'), predicted_match=(g in mapped_b) if cls != 'newly_appearing' else any(gf == g for _, gf in pr['pairs']), match_correct=mc,
                                 reference_change=CHX(ref['gchg'].get(g)), predicted_change=pr['change'].get((g, g)) if cls == 'persistent' else None, burden_reference=X.BU[ref['gburden']], burden_prediction=pr['burden'],
                                 lesion_size=size, distance=reg_d.get(g, np.nan), registration_quality=regq, tags=';'.join(tags), failure_category=';'.join(tags) if not mc else ''))
    df = pd.DataFrame(rows); df.to_csv(BASE / 'external_failure_matrix.csv', index=False)
    tags = ['small_lesion', 'large_lesion', 'close_neighbouring_lesion', 'disappeared_lesion', 'apparent_new_lesion', 'large_size_change', 'registration_sensitive', 'high_lesion_count']; out = []
    for name, g in df.groupby('method'):
        out.append(dict(method=name, tag='all', n_lesions=len(g), matching_error_rate=1 - g.match_correct.mean()))
        for t in tags:
            x = g[g.tags.fillna('').str.contains(t)]
            if len(x): out.append(dict(method=name, tag=t, n_lesions=len(x), matching_error_rate=1 - x.match_correct.mean()))
    pd.DataFrame(out).to_csv(BASE / 'external_failure_summary.csv', index=False)


def CHX(v): return X.CH.get(v) if v else None


def main_table(met, rep_df, names):
    rows = []; cons = pd.read_csv(BASE / 'external_consistency.csv') if (BASE / 'external_consistency.csv').exists() and (BASE / 'external_consistency.csv').stat().st_size > 5 else pd.DataFrame()
    ag = [r for r in read_jsonl(BASE / 'generic_agent_v2_raw.jsonl') if r.get('api_success') and r['repeat_id'] == 0]; vl = [r for r in read_jsonl(BASE / 'direct_vlm_raw.jsonl') if r.get('api_success') and r['repeat_id'] == 0]
    mv = [r for r in read_jsonl(BASE / 'medical_vlm_raw.jsonl') if r.get('run_success')]
    for n in names:
        g = lambda k: met[(met.method == n) & (met.metric == k)].iloc[0]
        f = lambda k: f'{g(k).estimate:.3f} [{g(k).ci95_lo:.3f}, {g(k).ci95_hi:.3f}]'
        cc = cons[cons.method == n].fraction_patients_fully_consistent.iloc[0] if len(cons) and (cons.method == n).any() else np.nan
        if n == AGENT2: tools, llm = (np.mean([r['n_tool_calls'] for r in ag]), np.mean([r['n_turns'] for r in ag])) if ag else (np.nan, np.nan)
        elif n in VLM_MODELS.values():
            mine = [r for r in vl if VLM_MODELS[r['model_id']] == n]; tools, llm = 0, (np.mean([2 if r.get('schema_repair_used') else 1 for r in mine]) if mine else np.nan)
        elif n in MEDVLM_MODELS.values():
            mine = [r for r in mv if MEDVLM_MODELS[r['model_key']] == n]; tools, llm = 0, (np.mean([2 if r.get('schema_repair_used') else 1 for r in mine]) if mine else np.nan)
        elif 'Full' in n: tools, llm = 4, 0
        else: tools, llm = (4, 0)
        cons_label = 'n/a (deterministic)' if n in DET.values() else ('n/a (greedy decoding, single run)' if n in MEDVLM_MODELS.values() else (f'{cc:.3f}' if cc == cc else 'not yet available'))
        rows.append({'Method': n, 'N evaluated': f'{int(g("match_f1").n_evaluated)}/{int(g("match_f1").n_planned)}', 'Matching F1': f('match_f1'), 'Lesion-change accuracy': f('change_accuracy'), 'Burden accuracy': f('burden_accuracy'),
                     'Conditional success': f'{g("conditional_task_success").estimate:.3f}', 'API coverage': f'{g("api_coverage").estimate:.3f}', 'Repeat consistency (fully consistent patients)': cons_label,
                     'Tool calls': tools, 'LLM calls': llm, 'Role': 'upper control (not deployable)' if 'oracle' in n else 'supplementary' if 'IoU' in n else 'primary'})
    pd.DataFrame(rows).to_csv(BASE / 'exp22_external_main_table.csv', index=False)


if __name__ == '__main__':
    main()
