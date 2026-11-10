#!/usr/bin/env python3
"""Sanity checks 1-10 (task section 18). --phase pre runs before any full system run (STOP if anything fails); --phase post checks the
logs of the completed runs. Results are merged into outputs/multivisit_longitudinal/sanity_checks.json.
  python sanity_checks.py --phase pre|post
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent / 'exp28_repeated_synchronization'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP28))
import agent_adapter as AA  # noqa: E402
import mu_glioma_adapter as A  # noqa: E402
import queries as Q  # noqa: E402
import timeline as T  # noqa: E402

OUT = T.OUT
EXP28_FILES = ['state/synchronization.py', 'state/versioning.py', 'state/patient_state.py', 'state/schemas.py', 'state/provenance.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pre():
    res = {}
    cfg = (OUT / 'experiment_config.yaml').read_text()
    coh = T.cohort()
    A.load_measure_cache(OUT / 'measurements_frozen.json')
    # 1 ordering
    bad = []
    for pid, v in coh.items():
        idx = [A.tp_index(x) for x in v]
        days = [A.mri_day(pid, x) for x in v if A.mri_day(pid, x) is not None]
        if idx != sorted(idx) or len(set(idx)) != len(idx) or any(b <= a for a, b in zip(days, days[1:])):
            bad.append(pid)
    res['1_patients_ordered_correctly'] = dict(passed=not bad, n_patients=len(coh), violations=bad)
    # 2 + 8 identical evidence: agent tool path vs OncoState observation path vs frozen reference, every visit
    G, GSA = AA.install(OUT / 'measurements_frozen.json')
    import run_oncostate as RO
    ref = pd.read_csv(OUT / 'reference_trajectories.csv').set_index(['patient_id', 'visit_id'])
    mism = []
    for pid, v in coh.items():
        A.VisitClock.set(pid, v, v, v[-1])
        tb = G.Toolbox()
        src = RO.MUSource()
        for tp in v:
            t = json.loads(tb.run('get_visit_record', {'patient_id': pid, 'timepoint': tp}))['result']
            o = src.observation(pid, tp)
            same = (t['measurements'] == {k: o['fields'][k]['value'] for k in t['measurements']}
                    and t['treatments_reported_since_previous_visit'] == o['fields']['treatments_reported']['value']
                    and t['days_from_diagnosis'] == o['fields']['days_from_diagnosis']['value']
                    and abs(t['measurements']['burden_ml'] - ref.loc[(pid, tp), 'burden_ml']) < 1e-9)
            if not same:
                mism.append(f'{pid}/{tp}')
    res['2_identical_visit_evidence_all_systems'] = dict(passed=not mism, n_visits=sum(map(len, coh.values())), mismatches=mism)
    res['8_current_state_measurements_agree'] = dict(passed=not mism, note='agent tool output == OncoState observation == frozen reference burden for every visit')
    # 3 leakage guard (negative tests)
    pid, v = next((p, x) for p, x in coh.items() if len(x) >= 4)
    A.VisitClock.set(pid, v, v[:2], v[1])
    tb = G.Toolbox()
    fut = json.loads(tb.run('get_visit_record', {'patient_id': pid, 'timepoint': v[2]}))
    lst = json.loads(tb.run('list_visits', {'patient_id': pid}))['result']['visits']
    other = json.loads(tb.run('get_visit_record', {'patient_id': next(p for p in coh if p != pid), 'timepoint': 'Timepoint_1'}))
    try:
        RO.MUSource().observation(pid, v[2]); onco_blocked = False
    except A.FutureAccess:
        onco_blocked = True
    res['3_future_access_blocked'] = dict(passed=(not fut['ok']) and 'FutureAccess' in fut['error'] and [x['visit'] for x in lst] == v[:2]
                                          and not other['ok'] and onco_blocked,
                                          tool_future_read=fut, list_visits=lst, other_patient_read_blocked=not other['ok'], oncostate_future_read_blocked=onco_blocked)
    # 6 official OncoState implementation
    import state.synchronization as SS
    import state.versioning as SV
    res['6_official_oncostate_implementation'] = dict(
        passed=Path(SS.__file__).resolve() == (EXP28 / 'state/synchronization.py').resolve() and Path(SV.__file__).resolve() == (EXP28 / 'state/versioning.py').resolve()
        and RO.synchronize is SS.synchronize and RO.persistent_step is SS.persistent_step,
        files={f: sha(EXP28 / f) for f in EXP28_FILES}, note='imported from experiments/exp28_repeated_synchronization/state, not copied or modified')
    # 7 reference determinism
    r1 = {(p, d): Q.dumps(Q.reference_answers([(x, A.measurement(p, x)[0]['burden_ml']) for x in v[:d]],
                                               {x: A.interval_treatments(p, v, x) for x in v[:d]})) for p, v in coh.items() for d in range(1, len(v) + 1)}
    m2 = {}
    for p, v in coh.items():                                       # independent re-measurement from the masks (no cache)
        for x in v:
            m2[(p, x)] = A.measure_mask(A.mask_path(p, x))[0]['burden_ml']
    r2 = {(p, d): Q.dumps(Q.reference_answers([(x, m2[(p, x)]) for x in v[:d]], {x: A.interval_treatments(p, v, x) for x in v[:d]}))
          for p, v in coh.items() for d in range(1, len(v) + 1)}
    rt = pd.read_csv(OUT / 'reference_trajectories.csv')
    r0 = {(r.patient_id, r.visit_depth): r.reference_answers for r in rt.itertuples()}
    ref_sha = sha(OUT / 'reference_trajectories.csv')
    res['7_reference_deterministic'] = dict(passed=r1 == r2 == r0 and ref_sha in cfg, reference_sha256=ref_sha, sha_matches_frozen_config=ref_sha in cfg,
                                            note='recomputed twice (cached and fresh mask re-measurement) == frozen file')
    # 9 no patient dropped
    res['9_no_patient_dropped_for_performance'] = dict(passed=len(coh) == 109 and '109' in cfg, n_included=len(coh),
                                                       note='cohort fixed by cohort_manifest.csv before any system run; checked again post-run')
    # 10 Exp 1 generic stateful agent unchanged
    res['10_exp1_generic_stateful_unchanged'] = dict(passed=all(AA.file_sha(k) == h for k, h in AA.EXP1_SHA256.items()) and GSA.__file__.endswith('stateful_baseline/generic_stateful_agent.py'),
                                                     files=AA.EXP1_SHA256, imported_from=GSA.__file__,
                                                     functions_used=['generic_stateful_agent.new_memory', 'generic_stateful_agent.episode', 'generic_stateful_agent.update_memory'])
    return res


def post():
    import analyze as AN
    res = {}
    coh = T.cohort()
    A.load_measure_cache(OUT / 'measurements_frozen.json')
    raws = {s: AN.load_agent(OUT / f'raw/{s}_natural.jsonl') for s in ('stateless', 'stateful', 'onco_llm')}
    raws['stateful_late'] = AN.load_agent(OUT / 'raw/stateful_late.jsonl')
    # 3 no future access in any log
    leak = {}
    for s, df in raws.items():
        if len(df):
            sc = AN.score_frame(df, coh)
            leak[s] = dict(successful_future_reads=int(df.future_successful_reads.sum()), blocked_future_attempts=int(df.future_blocked.sum()),
                           answers_mentioning_unavailable_visit=int(sc.future_mention.sum()))
    vers_bad = []
    for mode in ('natural', 'late'):
        root = OUT / f'oncostate_states/{mode}_run1'
        raw = [json.loads(l) for l in (OUT / f'raw/oncostate_{mode}_run1.jsonl').read_text().splitlines()]
        for r in raw:
            if r['query'] == '_sync' and not set(r['history_keys']) <= set(r['available']):
                vers_bad.append((mode, r['patient_id'], r['step_index']))
            if r['query'] == '_sync' and r['future_blocked']:
                vers_bad.append((mode, r['patient_id'], r['step_index'], 'blocked'))
        for pdir in root.iterdir():
            lin = json.loads((pdir / 'lineage.json').read_text())
            if not lin['versions']:
                vers_bad.append((mode, pdir.name, 'empty'))
    res['3_no_future_access_in_runs'] = dict(passed=all(v['successful_future_reads'] == 0 for v in leak.values()) and not vers_bad,
                                             agents=leak, oncostate_violations=vers_bad,
                                             note='answers mentioning an unavailable visit id are scored under consistency C4; none can stem from a read')
    # 4 generic stateful persists memory
    st = raws['stateful']
    grow = []
    for (p, r), g in st.groupby(['patient_id', 'repeat_id']):
        g = g.sort_values(['step_index'])
        first = g.groupby('step_index').memory_bytes_in.min()
        grow.append(bool((first.diff().dropna() > 0).all()))
    mem_visits = st[st.visit_depth >= 2].apply(lambda r: set(r.available[:-1]) <= set(r.memory_visits_after or []) if r.query == 'Q8' else None, axis=1).dropna()
    res['4_generic_stateful_persists_memory'] = dict(passed=bool(len(grow)) and all(grow) and st.memory_bytes_in.min() > 0,
                                                     sequences_with_growing_memory=f'{sum(grow)}/{len(grow)}',
                                                     q8_episodes_with_all_earlier_visit_records_in_memory=f'{int(mem_visits.sum())}/{len(mem_visits)}',
                                                     note='memory_bytes_in = size of the memory JSON placed in the user message (Exp 1 field)')
    # 5 stateless has no persistence
    sl = raws['stateless']
    lon = sl[(sl.visit_depth >= 2) & sl['query'].isin(['Q2', 'Q8'])]
    reread = lon.apply(lambda r: any(v != r.current_visit for v in r.source_reads), axis=1)
    res['5_stateless_no_persistence'] = dict(passed=sl.memory_bytes_in.isna().all(), episodes_with_memory=int(sl.memory_bytes_in.notna().sum()),
                                             q2_q8_episodes_rereading_earlier_visits=f'{int(reread.sum())}/{len(lon)}',
                                             note='every stateless episode is exp22b_agent.episode with a fresh Toolbox and no memory block')
    # 9 every patient present in every system
    o = pd.read_json(OUT / 'raw/oncostate_natural_run1.jsonl', lines=True)
    pres = {s: sorted(set(coh) - set(df.patient_id)) for s, df in raws.items() if len(df) and s != 'stateful_late'}
    pres['oncostate'] = sorted(set(coh) - set(o.patient_id))
    reps = {s: df.groupby('patient_id').repeat_id.nunique().value_counts().to_dict() for s, df in raws.items() if len(df)}
    res['9_no_patient_dropped'] = dict(passed=all(not v for v in pres.values()), missing_patients=pres, repeats_per_patient=reps)
    # OncoState determinism (run1 vs run2)
    a1 = (OUT / 'raw/oncostate_natural_run1.jsonl').read_text().splitlines()
    a2 = (OUT / 'raw/oncostate_natural_run2.jsonl').read_text().splitlines() if (OUT / 'raw/oncostate_natural_run2.jsonl').exists() else []
    strip = lambda l: {k: v for k, v in json.loads(l).items() if k not in ('sync_sec', 'query_sec', 'uncached_compute_sec')}  # noqa: E731
    same_ans = len(a1) == len(a2) and all(strip(x) == strip(y) for x, y in zip(a1, a2))
    h = lambda root: {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('v*.json'))}  # noqa: E731
    same_states = h(OUT / 'oncostate_states/natural_run1') == h(OUT / 'oncostate_states/natural_run2') if a2 else False
    res['6b_oncostate_deterministic'] = dict(passed=same_ans and same_states, identical_answers=same_ans, identical_state_files=same_states)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', required=True, choices=['pre', 'post']); a = ap.parse_args()
    f = OUT / 'sanity_checks.json'
    allres = json.loads(f.read_text()) if f.exists() else {}
    res = pre() if a.phase == 'pre' else post()
    allres[a.phase] = res
    allres[f'{a.phase}_all_passed'] = all(v['passed'] for v in res.values())
    f.write_text(json.dumps(allres, indent=1, default=str))
    for k, v in res.items():
        print(('PASS ' if v['passed'] else 'FAIL ') + k)
    if not allres[f'{a.phase}_all_passed']:
        sys.exit('SANITY CHECK FAILED - STOP')


if __name__ == '__main__':
    main()
