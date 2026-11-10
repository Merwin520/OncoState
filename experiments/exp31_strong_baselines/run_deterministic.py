#!/usr/bin/env python3
"""Deterministic systems of Exp 3.1: event-sourced clinical record (A: natural, late; B: revision), OncoState and non-durable revision
(B). Output records have the format of the multi-visit OncoState runner.   python run_deterministic.py"""
import copy
import json
import sys

from common import MV, OUT, A, Q, cohort, install_revision, steps

sys.path.insert(0, str(MV.parent / 'exp28_repeated_synchronization'))
import run_oncostate as RO  # noqa: E402  (MUSource, read_answers; read-only)
from state.synchronization import persistent_step, revise  # noqa: E402  (official, unchanged)
from state.versioning import VersionStore  # noqa: E402


def es_answer(log, visits, cur, asked):
    """Event-sourced: replay the append-only log; latest event per visit wins; chronological visit order."""
    view = {}
    for e in log:
        view[e['visit']] = e['record']
    seen = [v for v in visits if v in view and visits.index(v) <= visits.index(cur)]
    ans = Q.reference_answers([(v, view[v]['measurements']['burden_ml']) for v in seen],
                              {v: view[v]['treatments_reported_since_previous_visit'] for v in seen}) if seen and seen[-1] == cur else {}
    return {q: ans.get(q) for q in asked}


def rec(system, mode, pid, i, cur, avail, tag, q, ans):
    return dict(system=system, mode=mode, patient_id=pid, repeat_id=0, step_index=i, current_visit=cur, available=avail, tag=tag,
                visit_depth=len(avail), query=q, structured_answer=ans, success=ans is not None,
                failure_reason=None if ans is not None else 'not_answerable')


def eventsourced(pid, visits, mode):
    log, out = [], []
    for i, (cur, avail, tag) in enumerate(steps(visits, mode)):
        A.VisitClock.set(pid, visits, avail, cur, [])
        arrived = [v for v in avail if v not in {e['visit'] for e in log}]
        for v in arrived:
            log.append(dict(type='late_arrival' if tag.startswith('late:') else 'arrival', visit=v, record=A.visit_record(pid, v, visits)[0],
                            provenance=dict(source='released_reference_segmentation', step=i)))
        if tag.startswith('rev:'):
            v = tag[4:]
            log.append(dict(type='revision', visit=v, record=A.visit_record(pid, v, visits)[0], provenance=dict(source='revised record', step=i)))
        asked = T.queries_for(pid, avail, cur)
        ans = es_answer(log, visits, cur, asked)
        out += [rec('eventsourced', mode, pid, i, cur, avail, tag, q, ans[q]) for q in asked]
    return out


def oncostate_revision(pid, visits, store, durable=True):
    src, out = RO.MUSource(), []
    cfg = dict(patient_id=pid, timepoints=visits, phase_names={}, measure_keys=RO.MEASURE_KEYS)
    for i, (cur, avail, tag) in enumerate(steps(visits, 'revision')):
        A.VisitClock.set(pid, visits, avail, cur, [])
        view_patch = None
        if tag.startswith('rev:'):
            v = tag[4:]
            corrected = A.visit_record(pid, v, visits)[0]['measurements']
            s = store.latest(pid)
            for k in ('burden_ml', 'et_ml', 'netc_ml', 'snfh_ml'):
                s, r = revise(s, v, k, corrected[k], reason='revised record', actor='radiology report', source='revised record', cfg=cfg,
                              now=f'revision-step{i}')
                if durable:
                    store.write(s, dict(event_type='revision'))
            view_patch = (v, corrected)
        persistent_step(store, src, pid, cur, cfg, now=f'revision-step{i}')
        latest = store.latest(pid)
        if not durable and view_patch:                       # corrected only in the answers of this review step
            latest = copy.deepcopy(latest)
            for k in ('burden_ml', 'et_ml', 'netc_ml', 'snfh_ml'):
                latest['history'][view_patch[0]]['fields'][k]['value'] = view_patch[1][k]
        asked = T.queries_for(pid, avail, cur)
        ans = RO.read_answers(latest, asked)
        out += [rec('oncostate' if durable else 'nondurable', 'revision', pid, i, cur, avail, tag, q, ans[q]) for q in asked]
    assert store.verify(pid)
    return out


T = __import__('timeline')


def main():
    A.load_measure_cache(MV.parents[1] / 'outputs/multivisit_longitudinal/measurements_frozen.json')
    raw = OUT / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    for mode, mv in (('natural', 0), ('late', 4)):
        with open(raw / f'eventsourced_{mode}.jsonl', 'w') as f:
            for pid, v in cohort(mv).items():
                for r in eventsourced(pid, v, mode):
                    f.write(json.dumps(r, default=str) + '\n')
    install_revision()
    with open(raw / 'eventsourced_revision.jsonl', 'w') as f:
        for pid, v in cohort(4).items():
            for r in eventsourced(pid, v, 'revision'):
                f.write(json.dumps(r, default=str) + '\n')
    for durable, name in ((True, 'oncostate'), (False, 'nondurable')):
        store = VersionStore(OUT / f'states_{name}_revision')
        with open(raw / f'{name}_revision.jsonl', 'w') as f:
            for pid, v in cohort(4).items():
                for r in oncostate_revision(pid, v, store, durable):
                    f.write(json.dumps(r, default=str) + '\n')
    print('deterministic systems written')


if __name__ == '__main__':
    main()
