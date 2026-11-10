#!/usr/bin/env python3
"""System C: Full OncoState on MU-Glioma-Post, using the official Exp 2.8 state code UNCHANGED
(exp28_repeated_synchronization/state/synchronization.py: synchronize, persistent_step; state/versioning.py: VersionStore).

Adapter: MUSource (EvidenceSource interface: clinical(), observation(), .counter) turning the shared visit record into an observation
record with field-level provenance. Queries are answered by a deterministic reader over the LATEST STORED state version only.
  python run_oncostate.py --mode natural|late --run run1
Outputs: oncostate_states/<mode>_<run>/<patient>/v*.json + lineage.json, raw/oncostate_<mode>_<run>.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent / 'exp28_repeated_synchronization'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP28))
import mu_glioma_adapter as A  # noqa: E402
import queries as Q  # noqa: E402
import timeline as T  # noqa: E402
from state.synchronization import Counter, persistent_step, synchronize  # noqa: E402  (official, unchanged)
from state.versioning import VersionStore  # noqa: E402  (official, unchanged)

OUT = T.OUT
MEASURE_KEYS = ['burden_ml', 'et_ml', 'netc_ml', 'snfh_ml', 'resection_cavity_ml']


class MUSource:
    """EvidenceSource for exp28 synchronize(): every read goes through the VisitClock (future visits raise FutureAccess)."""
    def __init__(self):
        self.counter = Counter()
        self.compute = 0.0

    def clinical(self, pid):
        self.counter.source_reads += 1
        return {'fields': {}, 'provenance': dict(source_type='none', note='no baseline clinical fields are given to any system')}

    def observation(self, pid, tp):
        ctx = A.VisitClock.check(pid, tp)
        rec, secs = A.visit_record(pid, tp, ctx['visits'])
        self.counter.source_reads += 1; self.counter.tool_calls += 1; self.compute += secs
        prov = dict(source_type='released_reference_segmentation', source_file=rec['source']['file'], source_study_uid=None, source_series_uid=None,
                    extraction_version=rec['source']['measurement_version'], processing_version='exp28-sync-1.0')
        fields = {k: dict(value=v, unit='mL', available=v is not None, provenance=prov, qc_flags=[]) for k, v in rec['measurements'].items()}
        fields['days_from_diagnosis'] = dict(value=rec['days_from_diagnosis'], unit='days', available=rec['days_from_diagnosis'] is not None,
                                             provenance=dict(prov, source_type='clinical_spreadsheet_mri_day'), qc_flags=[])
        fields['treatments_reported'] = dict(value=rec['treatments_reported_since_previous_visit'], unit=None,
                                             available=rec['treatments_reported_since_previous_visit'] is not None,
                                             provenance=dict(prov, source_type='clinical_spreadsheet_treatment_days'), qc_flags=[])
        return dict(status='observed', study_uid=f'{pid}/{tp}', series_uids=[rec['source']['file']], fields=fields, qc_flags=[])


def read_answers(state, asked):
    """Deterministic query reader: applies the frozen query rules (queries.reference_answers) to the state's own history."""
    order, cur = state['protocol_timepoints'], state['current_timepoint']
    seen = [tp for tp in order[:order.index(cur) + 1] if state['observation_status'].get(tp) in ('observed', 'late_arrival')]
    seq = [(tp, state['history'][tp]['fields']['burden_ml']['value']) for tp in seen]
    tre = {tp: state['history'][tp]['fields']['treatments_reported']['value'] for tp in seen}
    ans = Q.reference_answers(seq, tre) if seq and seq[-1][0] == cur else {}
    return {q: ans.get(q) for q in asked}


def run_patient(pid, visits, mode, store):
    src, recs = MUSource(), []
    cfg = dict(patient_id=pid, timepoints=visits, phase_names={}, measure_keys=MEASURE_KEYS)
    for i, (cur, avail, tag) in enumerate(T.steps(visits, mode)):
        log = []
        A.VisitClock.set(pid, visits, avail, cur, log)
        t0 = time.perf_counter(); src.compute = 0.0
        if tag.startswith('late:'):                     # explicit late-arrival event (official synchronize, event_type='late_arrival')
            c = Counter(); src.counter = c
            prev = store.latest(pid); c.state_reads += 1
            s, ev = synchronize(prev, tag[5:], src.observation(pid, tag[5:]), cfg=cfg, event_type='late_arrival', now=f'{mode}-step{i}')
            store.write(s, ev); c.state_ops += 1
            ev.update(c.snapshot(), strategy='persistent')
        elif mode == 'late' and i == 1:                  # V2 hidden: recorded explicitly as hidden_stress_test, then V3 scheduled
            c = Counter(); src.counter = c
            prev = store.latest(pid); c.state_reads += 1
            hidden = visits[1]
            s, ev0 = synchronize(prev, hidden, {'status': 'hidden_stress_test', 'reason': 'withheld for late-arrival stress test'}, cfg=cfg,
                                 now=f'{mode}-step{i}-hidden')
            store.write(s, ev0)
            s, ev = synchronize(s, cur, src.observation(pid, cur), cfg=cfg, now=f'{mode}-step{i}')
            store.write(s, ev); c.state_ops += 2
            ev.update(c.snapshot(), strategy='persistent')
        else:
            s, ev = persistent_step(store, src, pid, cur, cfg, now=f'{mode}-step{i}')
        sync_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        latest = store.latest(pid)                      # queries are answered from the stored version only
        asked = T.queries_for(pid, avail, cur)
        ans = read_answers(latest, asked)
        q_sec = time.perf_counter() - t1
        for q in asked:
            recs.append(dict(system='oncostate', mode=mode, patient_id=pid, repeat_id=0, step_index=i, current_visit=cur, available=avail, tag=tag,
                             visit_depth=len(avail), query=q, structured_answer=ans[q], success=ans[q] is not None,
                             failure_reason=None if ans[q] is not None else 'not_answerable_from_state'))
        recs.append(dict(system='oncostate', mode=mode, patient_id=pid, repeat_id=0, step_index=i, current_visit=cur, available=avail, tag=tag,
                         visit_depth=len(avail), query='_sync', state_version=latest['state_version'], source_reads=ev['source_reads'],
                         tool_calls=ev['tool_calls'], state_reads=ev['state_reads'] + 1, state_ops=ev['state_ops'], llm_calls=0,
                         uncached_compute_sec=round(src.compute, 4), sync_sec=sync_sec, query_sec=q_sec,
                         visits_read=[x['visit'] for x in log if x['allowed']], future_blocked=sum(not x['allowed'] for x in log),
                         history_keys=sorted(latest['history']), observation_status=latest['observation_status']))
    return recs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='natural'); ap.add_argument('--run', default='run1')
    ap.add_argument('--pilot', type=int, default=0); a = ap.parse_args()
    coh = T.cohort()
    if a.mode == 'late':
        coh = {p: v for p, v in coh.items() if len(v) >= 4}
    if a.pilot:
        coh = dict(list(coh.items())[:a.pilot])
    A.load_measure_cache(OUT / 'measurements_frozen.json')
    sub = 'pilot' if a.pilot else ''
    root = OUT / sub / f'oncostate_states/{a.mode}_{a.run}'
    raw = OUT / sub / f'raw/oncostate_{a.mode}_{a.run}.jsonl'
    if raw.exists() or root.exists():
        raise SystemExit(f'{raw} or {root} exists; refusing to overwrite')
    raw.parent.mkdir(parents=True, exist_ok=True)
    store = VersionStore(root)
    t0 = time.time()
    with open(raw, 'w') as f:
        for pid, visits in coh.items():
            for r in run_patient(pid, visits, a.mode, store):
                f.write(json.dumps(r, default=str) + '\n')
            assert store.verify(pid), f'lineage broken for {pid}'
    print('OncoState', a.mode, a.run, 'patients', len(coh), 'seconds', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
