"""I-SPY1 spreadsheet-only external replication using the same synchronization engine."""
import json

import pandas as pd

from evaluation.continuity import state_checks
from features.measurement_features import ISPY1EvidenceSource, TP1, ispy1_cfg
from paths import EXPORTS, work_dir
from state.synchronization import persistent_step, stateless_rebuild
from state.versioning import VersionStore


def run():
    source = ISPY1EvidenceSource()
    store = VersionStore(work_dir() / 'ispy1_states')
    events_path = work_dir() / 'ispy1_events.jsonl'
    efficiency_path = work_dir() / 'ispy1_efficiency.jsonl'
    done = set()
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            event = json.loads(line); done.add((str(event['patient_id']), event['timepoint']))
    for number, patient_id in enumerate(sorted(source.data.index.astype(str)), 1):
        cfg = ispy1_cfg(patient_id)
        for index, timepoint in enumerate(TP1):
            if (patient_id, timepoint) in done:
                continue
            state, event = persistent_step(store, source, patient_id, timepoint, cfg, now=f'external-{timepoint}')
            rebuilt, stateless = stateless_rebuild(source, patient_id, timepoint, cfg, now=f'external-{timepoint}')
            assert state['history'] == rebuilt['history']
            with events_path.open('a') as handle:
                handle.write(json.dumps(event, default=str) + '\n')
            keys = ['source_reads', 'series_reads', 'tool_calls', 'state_reads', 'state_ops', 'runtime_seconds']
            with efficiency_path.open('a') as handle:
                handle.write(json.dumps(dict(patient_id=patient_id, timepoint=timepoint, n_prior_visits=index,
                                             strategy='persistent', **{key: event[key] for key in keys})) + '\n')
                handle.write(json.dumps(dict(n_prior_visits=index, **stateless)) + '\n')
        if number % 50 == 0:
            print(f'I-SPY1 synchronized {number}/{len(source.data)}', flush=True)
    rows = []
    for patient_id in sorted(source.data.index.astype(str)):
        states = store.all_versions(patient_id)
        assert len(states) == 4 and store.verify(patient_id)
        final = states[-1]
        previous = None
        for state in states:
            checks = state_checks(state, previous)
            rows.append(dict(patient_id=patient_id, timepoint=state['current_timepoint'],
                             state_version=state['state_version'], parent_version=state['parent_version'],
                             observed=state['current_timepoint'] in state['history'],
                             history_retention_rate=checks['history_retention_rate'],
                             provenance_complete=checks['provenance_complete'],
                             historical_mutation_detected=checks['historical_mutation_detected'],
                             lineage_integrity=True))
            previous = state
        for timepoint in TP1:
            observation = source.observation(patient_id, timepoint)
            record = final['history'].get(timepoint)
            reference = (observation or {}).get('fields', {}).get('longest_diameter_mm', {}).get('value')
            value = (record or {}).get('fields', {}).get('longest_diameter_mm', {}).get('value')
            rows.append(dict(patient_id=patient_id, timepoint=timepoint, state_version='final_check',
                             parent_version=None, observed=reference is not None,
                             history_retention_rate=None, provenance_complete=True,
                             historical_mutation_detected=False, lineage_integrity=True,
                             reference_value=reference, state_value=value,
                             absolute_error=None if reference is None or value is None else abs(reference - value)))
    pd.DataFrame(rows).to_csv(EXPORTS / 'ispy1_external_replication.csv', index=False)
    efficiency = pd.DataFrame([json.loads(line) for line in efficiency_path.read_text().splitlines()])
    efficiency.drop_duplicates(['patient_id', 'timepoint', 'strategy'], keep='last').to_csv(
        EXPORTS / 'ispy1_sync_efficiency.csv', index=False)
    pd.DataFrame([dict(patient_id=patient_id, **source.outcome(patient_id)) for patient_id in source.data.index]).to_csv(
        EXPORTS / 'ispy1_outcomes_separate.csv', index=False)
    print(f'I-SPY1 complete: {len(source.data)} patients')


if __name__ == '__main__':
    run()
