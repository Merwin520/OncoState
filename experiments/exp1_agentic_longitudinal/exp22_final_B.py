"""Experiment 2.2-B: persistent vs ephemeral patient state (functional ablation) + state-edit retention check.

Both conditions use the SAME per-timepoint lesion masks, the SAME registration, the SAME matcher
(exp22_run.match_fast) and the SAME LLM-written query programs (exp22 llm_generated_code.json, sample 0;
no new LLM calls). The only difference is whether the longitudinal state is built once and kept, or rebuilt
from the raw per-timepoint outputs for every query.

Part 1: query ablation on S0 (primary) and A (sensitivity).
Part 2: state-edit check with GT-derived *simulated* verified corrections (engineering validation only).
"""
import json, sys, tempfile, time, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

import exp22_run as R
import exp22_llm as L
import exp22_eval as E

ROOT = Path(__file__).resolve().parent
DST = ROOT / 'outputs/exp22_persistent_state_final'
DST.mkdir(parents=True, exist_ok=True)
SETTINGS = ['S0', 'A']
QS = list(L.QUERIES)
PAIRS = {p['patient_id']: p for p in json.loads((R.OUT / 'pairs.json').read_text())}
CODES = {q: v[0]['code'] for q, v in json.loads((R.OUT / 'llm_generated_code.json').read_text()).items()}
MAX_EDITS_PER_PATIENT = 3


def rebuild(setting, pid):
    """Full longitudinal reconstruction from raw per-timepoint lesion masks (registration is the cached .tfm)."""
    p = PAIRS[pid]
    tx = R.register(p['baseline_ct'], p['followup_ct'], R.OUT / f'reg/{pid}.tfm')
    bl = sitk.ReadImage(str(R.PATHS[setting](pid, 'baseline'))); fu = sitk.ReadImage(str(R.PATHS[setting](pid, 'followup')))
    rows, bd, fd = R.match_fast(bl, fu, tx)
    return {'patient_id': pid, 'setting': setting, 'status': 'completed', 'baseline_lesions': bd, 'followup_lesions': fd,
            'correspondences': rows, 'units': {'volume': 'mm3', 'diameter': 'mm', 'centroid': 'mm LPS'}}


def exec_code(q, state_path):
    t0 = time.monotonic()
    got, err = L.run_code(CODES[q], state_path)
    return got, err, time.monotonic() - t0


def answer_ids(q, ans):
    if q == 'Q0_inventory':
        return {x for r in ans for x in (r['baseline_lesion_id'], r['followup_lesion_id']) if x is not None}
    if q == 'Q1_increased_gt20':
        return {x for pr in ans for x in pr}
    if q == 'Q3_disappeared':
        return set(ans)
    return set()


def consistent(ans):
    """Cross-query consistency of the four answers about one patient."""
    q0, q1, q2, q3 = ans['Q0_inventory'], ans['Q1_increased_gt20'], ans['Q2_new_count'], ans['Q3_disappeared']
    if any(a is None for a in (q0, q1, q2, q3)):
        return False
    c1 = sum(1 for r in q0 if r['status'] == 'newly_appearing') == q2
    c2 = sorted(r['baseline_lesion_id'] for r in q0 if r['status'] == 'disappearing') == q3
    c3 = sorted([r['baseline_lesion_id'], r['followup_lesion_id']] for r in q0
                if r['status'] == 'persistent' and r['volume_change_pct'] is not None and r['volume_change_pct'] > 20) == sorted(q1)
    return bool(c1 and c2 and c3)


def part1_patient(args):
    setting, pid = args
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    persisted = R.OUT / f'state_{setting}/{pid}.json'
    ref_state = json.loads(persisted.read_text())
    refs = {q: L.reference(ref_state, q) for q in QS}
    rows, ans_by_cond = [], {}
    # ---- persistent: build once, keep, read for every query
    t0 = time.monotonic(); st = rebuild(setting, pid); t_build = time.monotonic() - t0
    rt = json.loads(json.dumps(st))
    key = lambda rr: sorted((r['baseline_lesion_id'] or -1, r['followup_lesion_id'] or -1, r['status']) for r in rr)
    same_as_persisted = (key(rt['correspondences']) == key(ref_state['correspondences']),   # ids + status
                         rt['correspondences'] == ref_state['correspondences'])              # incl. distance/cost evidence values
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / 'state.json'; sp.write_text(json.dumps(st))
        ans = {}
        for i, q in enumerate(QS):
            got, err, t = exec_code(q, sp)
            ans[q] = got
            ids = answer_ids(q, got) if got is not None else set()
            resolves = ids <= ({int(k) for k in st['baseline_lesions']} | {int(k) for k in st['followup_lesions']})
            rows.append(dict(patient_id=pid, setting=setting, condition='persistent', query=q, reconstructions=1 if i == 0 else 0,
                             seconds=t + (t_build if i == 0 else 0), executed=err is None,
                             correct_vs_reference=err is None and L.close(got, refs[q]),
                             provenance_retrievable=bool(resolves)))  # ids resolve in a state artifact that remains on disk
        ans_by_cond['persistent'] = ans
    # ---- ephemeral: rebuild from raw per-timepoint outputs for each query, discard afterwards
    ans = {}
    for q in QS:
        with tempfile.TemporaryDirectory() as td:
            t0 = time.monotonic(); st = rebuild(setting, pid); tb = time.monotonic() - t0
            sp = Path(td) / 'state.json'; sp.write_text(json.dumps(st))
            got, err, t = exec_code(q, sp)
        ans[q] = got
        rows.append(dict(patient_id=pid, setting=setting, condition='ephemeral', query=q, reconstructions=1, seconds=tb + t,
                         executed=err is None, correct_vs_reference=err is None and L.close(got, refs[q]),
                         provenance_retrievable=False))  # transient state deleted when the query returns
    ans_by_cond['ephemeral'] = ans
    cons = [dict(patient_id=pid, setting=setting, condition=c, cross_query_consistent=consistent(a)) for c, a in ans_by_cond.items()]
    return rows, cons, same_as_persisted


def apply_relink(state, pb, pf, etype):
    """Edit the persistent state: make (pb, pf) a persistent pair; orphans of removed rows fall back to disappearing/new."""
    rows = [dict(r) for r in state['correspondences']]
    removed = [r for r in rows if r['baseline_lesion_id'] == pb or r['followup_lesion_id'] == pf]
    keep = [r for r in rows if r not in removed]
    for r in removed:
        if r['status'] == 'persistent':
            if r['baseline_lesion_id'] != pb:
                keep.append({'baseline_lesion_id': r['baseline_lesion_id'], 'followup_lesion_id': None, 'status': 'disappearing'})
            if r['followup_lesion_id'] != pf:
                keep.append({'baseline_lesion_id': None, 'followup_lesion_id': r['followup_lesion_id'], 'status': 'newly_appearing'})
    keep.append({'baseline_lesion_id': pb, 'followup_lesion_id': pf, 'status': 'persistent', 'edited': True})
    new = json.loads(json.dumps(state))
    new['correspondences'] = keep
    new.setdefault('edit_log', []).append({'type': etype, 'baseline_lesion_id': pb, 'followup_lesion_id': pf, 'source': 'simulated_from_GT'})
    return new, removed


def part2_patient(args):
    setting, pid = args
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    st = json.loads((R.OUT / f'state_{setting}/{pid}.json').read_text())
    gb, gf = E.gt_img(pid, 'baseline'), E.gt_img(pid, 'followup')
    garr_b, garr_f = sitk.GetArrayFromImage(gb), sitk.GetArrayFromImage(gf)
    pa_b = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting](pid, 'baseline'))))
    pa_f = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting](pid, 'followup'))))
    mb, _, _, _ = E.overlap_map(pa_b, garr_b); mf, _, _, _ = E.overlap_map(pa_f, garr_f)
    inv_b = {g: p for p, g in mb.items()}; inv_f = {g: p for p, g in mf.items()}
    gbl = {int(v) for v in np.unique(garr_b) if v}; gfu = {int(v) for v in np.unique(garr_f) if v}
    cur = {(r['baseline_lesion_id'], r['followup_lesion_id']) for r in st['correspondences'] if r['status'] == 'persistent'}
    cand = []
    for g in sorted(gbl & gfu):
        pb, pf = inv_b.get(g), inv_f.get(g)
        if pb is None or pf is None or (pb, pf) in cur:
            continue
        rb = [r for r in st['correspondences'] if r['baseline_lesion_id'] == pb][0]
        rf = [r for r in st['correspondences'] if r['followup_lesion_id'] == pf][0]
        etype = 'missed_link' if rb['status'] == 'disappearing' and rf['status'] == 'newly_appearing' else 'wrong_link'
        cand.append((g, pb, pf, etype))
    ev_rows, q_rows = [], []
    ref_before = {q: L.reference(st, q) for q in QS}
    with tempfile.TemporaryDirectory() as td:
        orig_path = Path(td) / 'orig.json'; orig_path.write_text(json.dumps(st))
        for k, (g, pb, pf, etype) in enumerate(cand[:MAX_EDITS_PER_PATIENT]):
            cid = f'{setting}_{pid}_{k}'
            edited, removed = apply_relink(st, pb, pf, etype)
            ep = Path(td) / f'edited_{k}.json'; ep.write_text(json.dumps(edited))  # persisted, versioned state
            ev_rows.append(dict(patient_id=pid, setting=setting, correction_id=cid, correction_type=etype, gt_lesion_id=g,
                                baseline_lesion_id=pb, followup_lesion_id=pf, simulated_from_GT=True,
                                before_state=json.dumps([{kk: vv for kk, vv in r.items() if kk in ('baseline_lesion_id', 'followup_lesion_id', 'status')}
                                                         for r in removed]),
                                corrected_state=json.dumps({'baseline_lesion_id': pb, 'followup_lesion_id': pf, 'status': 'persistent'})))
            for q in QS:
                ref_after = L.reference(edited, q)
                affected = not L.close(ref_before[q], ref_after)
                # persistent: later query reads the edited state kept on disk
                got_p, err_p, _ = exec_code(q, ep)
                # ephemeral: the edit was not retained; later query rebuilds from raw outputs (verified identical to original state in part 1)
                got_e, err_e, _ = exec_code(q, orig_path)
                for strat, got, err in (('persistent', got_p, err_p), ('ephemeral', got_e, err_e)):
                    ok = err is None and L.close(got, ref_after)
                    stale = err is None and affected and L.close(got, ref_before[q])
                    q_rows.append(dict(patient_id=pid, setting=setting, correction_id=cid, strategy=strat, query_type=q, affected=affected,
                                       predicted=json.dumps(got)[:300], reference=json.dumps(ref_after)[:300], correct=ok, stale=stale))
    return ev_rows, q_rows, len(cand)


def main():
    jobs = [(s, pid) for s in SETTINGS for pid in PAIRS if (R.OUT / f'state_{s}/{pid}.json').exists()]
    with Pool(16) as pool:
        p1 = pool.map(part1_patient, jobs)
        p2 = pool.map(part2_patient, jobs)
    q_rows = pd.DataFrame([r for rows, _, _ in p1 for r in rows])
    cons = pd.DataFrame([c for _, cs, _ in p1 for c in cs])
    struct_same = sum(same[0] for _, _, same in p1); exact_same = sum(same[1] for _, _, same in p1)
    q_rows.to_csv(DST / 'ablation_query_results.csv', index=False)
    cons.to_csv(DST / 'ablation_consistency.csv', index=False)
    ev = pd.DataFrame([r for rows, _, _ in p2 for r in rows]); qq = pd.DataFrame([r for _, rows, _ in p2 for r in rows])
    ev.to_csv(DST / 'correction_events.csv', index=False); qq.to_csv(DST / 'correction_query_results.csv', index=False)

    # patient-level summary for the ablation
    summ = {'rebuild_vs_persisted_states': {'n_states': len(p1), 'ids_and_status_identical': int(struct_same), 'evidence_values_identical': int(exact_same)}, 'new_llm_calls': 0, 'n_patients': len({p for _, p in jobs}),
            'ablation': {}, 'edit_check': {}}
    for s in SETTINGS:
        for c in ('persistent', 'ephemeral'):
            d = q_rows[(q_rows.setting == s) & (q_rows.condition == c)]
            per_pat = d.groupby('patient_id').agg(recon=('reconstructions', 'sum'), sec=('seconds', 'sum'))
            summ['ablation'][f'{s}/{c}'] = dict(
                queries=len(d), answer_accuracy_vs_reference=float(d.correct_vs_reference.mean()), executed=float(d.executed.mean()),
                reconstructions_per_patient_4q=float(per_pat.recon.mean()), seconds_per_patient_4q=float(per_pat.sec.mean()),
                provenance_retrievable=float(d.provenance_retrievable.mean()),
                cross_query_consistency=float(cons[(cons.setting == s) & (cons.condition == c)].cross_query_consistent.mean()))
        e = qq[qq.setting == s]
        for strat in ('persistent', 'ephemeral'):
            d = e[e.strategy == strat]; aff = d[d.affected]
            summ['edit_check'][f'{s}/{strat}'] = dict(
                correction_events=int(d.correction_id.nunique()), query_instances=len(d), affected_query_instances=len(aff),
                retention_rate_affected=float(aff.correct.mean()) if len(aff) else None,
                stale_answers=int(d.stale.sum()), all_query_accuracy=float(d.correct.mean()) if len(d) else None,
                answers_updated_per_edit=float(aff.correct.sum() / max(d.correction_id.nunique(), 1)))
    summ['edit_events_by_type'] = ev.groupby(['setting', 'correction_type']).size().rename('n').reset_index().to_dict('records') if len(ev) else []
    summ['candidate_errors_available'] = {s: int(sum(n for (ss, _), (_, _, n) in zip(jobs, p2) if ss == s)) for s in SETTINGS}
    (DST / 'exp22_B_summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
