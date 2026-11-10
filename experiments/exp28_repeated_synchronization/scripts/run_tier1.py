"""Run full-cohort I-SPY2 feature-only synchronization and stateless comparator."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.continuity import state_checks  # noqa: E402
from features.measurement_features import ISPY2EvidenceSource, TP2, ispy2_cfg  # noqa: E402
from paths import EXPORTS, STATES  # noqa: E402
from state.synchronization import persistent_step, stateless_rebuild  # noqa: E402
from state.versioning import VersionStore  # noqa: E402


def append_jsonl(path, row):
    with path.open('a') as handle:
        handle.write(json.dumps(row, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    EXPORTS.mkdir(exist_ok=True)
    STATES.mkdir(exist_ok=True)
    source = ISPY2EvidenceSource()
    patients = sorted(source.visits.patient_id.unique())[:args.limit]
    store = VersionStore(STATES)
    events_file = EXPORTS / 'synchronization_events.jsonl'
    efficiency_file = EXPORTS / 'sync_efficiency_events.jsonl'
    done = set()
    if events_file.exists():
        for line in events_file.read_text().splitlines():
            event = json.loads(line)
            done.add((str(event['patient_id']), event['timepoint']))
    for number, patient_id in enumerate(patients, 1):
        cfg = ispy2_cfg(patient_id)
        for index, timepoint in enumerate(TP2):
            if (patient_id, timepoint) in done:
                continue
            state, event = persistent_step(store, source, patient_id, timepoint, cfg, now=f'protocol-{timepoint}')
            rebuilt, stateless = stateless_rebuild(source, patient_id, timepoint, cfg, now=f'protocol-{timepoint}')
            if state['history'] != rebuilt['history'] or state['observation_status'] != rebuilt['observation_status']:
                raise AssertionError(f'persistent/stateless evidence mismatch: {patient_id} {timepoint}')
            event['n_prior_visits'] = index
            append_jsonl(events_file, event)
            keys = ['strategy', 'source_reads', 'series_reads', 'tool_calls', 'state_reads', 'state_ops', 'runtime_seconds']
            append_jsonl(efficiency_file, dict(patient_id=patient_id, timepoint=timepoint,
                                               n_prior_visits=index, **{key: event[key] for key in keys}))
            append_jsonl(efficiency_file, dict(n_prior_visits=index, **stateless))
        if number % 50 == 0 or number == len(patients):
            print(f'synchronized {number}/{len(patients)} patients', flush=True)
    build_tables(patients, source, store, events_file, efficiency_file)


def build_tables(patients, source, store, events_file, efficiency_file):
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    event_map = {(str(event['patient_id']), event['timepoint']): event for event in events}
    versions, longitudinal, fidelity, outcomes = [], [], [], []
    for patient_id in patients:
        cfg = ispy2_cfg(patient_id)
        states = store.all_versions(patient_id)
        if len(states) != 4 or not store.verify(patient_id):
            raise AssertionError(f'incomplete/invalid lineage {patient_id}: {len(states)}')
        previous = None
        for state in states:
            timepoint = state['current_timepoint']
            event = event_map[(patient_id, timepoint)]
            checks = state_checks(state, previous)
            if checks['future_visit_visible'] or checks['forbidden_outcome_fields']:
                raise AssertionError(f'leakage in {patient_id} {timepoint}: {checks}')
            versions.append(dict(patient_id=patient_id, timepoint=timepoint,
                                 state_version=state['state_version'], parent_version=state['parent_version'],
                                 n_historical_observations=len(state['history']), n_new_fields=len(event['new_fields']),
                                 n_missing_fields=len(event['missing_fields']),
                                 provenance_complete=checks['provenance_complete'],
                                 historical_mutation_detected=checks['historical_mutation_detected'],
                                 history_retention_rate=checks['history_retention_rate'], lineage_integrity=True))
            previous = state
        final = states[-1]
        outcomes.append(source.outcome(patient_id))
        prior = {}
        baseline = {}
        for timepoint in TP2:
            observation = source.observation(patient_id, timepoint)
            record = final['history'].get(timepoint)
            ftv_reference = (observation or {}).get('fields', {}).get('ftv_pe_cc', {}).get('value')
            ftv_state = (record or {}).get('fields', {}).get('ftv_pe_cc', {}).get('value')
            ftv_prior = prior.get('ftv_pe_cc')
            delta = None if ftv_reference is None or ftv_prior is None else ftv_reference - ftv_prior[0]
            percent = None if ftv_reference is None or baseline.get('ftv_pe_cc') in (None, 0) else (
                100 * (ftv_reference - baseline['ftv_pe_cc']) / baseline['ftv_pe_cc'])
            longitudinal.append(dict(patient_id=patient_id, timepoint=timepoint, state_version=final['state_version'],
                                     observed_measurement=ftv_reference, state_measurement=ftv_state,
                                     observed_FTV=ftv_reference, state_FTV=ftv_state, delta_from_previous=delta,
                                     percent_change_from_T0=percent, treatment_phase=cfg['phase_names'][timepoint],
                                     missing_flag=observation is None,
                                     QC_flag=';'.join((observation or {}).get('qc_flags', [])),
                                     source_uid=(observation or {}).get('study_uid')))
            for measure in cfg['measure_keys']:
                reference = (observation or {}).get('fields', {}).get(measure, {}).get('value')
                value = (record or {}).get('fields', {}).get(measure, {}).get('value')
                if reference is None and value is None:
                    continue
                direction = correct = None
                if measure in prior and reference is not None and value is not None:
                    reference_direction = (reference > prior[measure][0]) - (reference < prior[measure][0])
                    state_direction = (value > prior[measure][1]) - (value < prior[measure][1])
                    direction = {-1: 'decrease', 0: 'stable', 1: 'increase'}[reference_direction]
                    correct = reference_direction == state_direction
                error = None if reference is None or value is None else abs(value - reference)
                fidelity.append(dict(patient_id=patient_id, timepoint=timepoint, measure=measure,
                                     reference_value=reference, state_value=value, absolute_error=error,
                                     relative_error=None if error is None or reference == 0 else error / abs(reference),
                                     direction_from_previous=direction, direction_correct=correct))
                if reference is not None and value is not None:
                    prior[measure] = (reference, value)
                    baseline.setdefault(measure, reference)
    pd.DataFrame(versions).to_csv(EXPORTS / 'state_versions.csv', index=False)
    pd.DataFrame(longitudinal).to_csv(EXPORTS / 'longitudinal_trajectory.csv', index=False)
    pd.DataFrame(fidelity).to_csv(EXPORTS / 'trajectory_fidelity.csv', index=False)
    pd.DataFrame(outcomes).to_csv(EXPORTS / 'ispy2_outcomes_separate.csv', index=False)
    efficiency = pd.DataFrame([json.loads(line) for line in efficiency_file.read_text().splitlines()])
    efficiency = efficiency.drop_duplicates(['patient_id', 'timepoint', 'strategy'], keep='last')
    efficiency.to_csv(EXPORTS / 'sync_efficiency.csv', index=False)
    natural = []
    for patient_id in patients:
        final = store.latest(patient_id)
        for timepoint in TP2:
            actually_missing = (patient_id, timepoint) not in source._vis.index
            recorded = final['observation_status'].get(timepoint) == 'not_observed'
            natural.append(dict(patient_id=patient_id, timepoint=timepoint, actually_missing=actually_missing,
                                missing_recorded_correctly=recorded == actually_missing,
                                silent_imputation=actually_missing and timepoint in final['history']))
    pd.DataFrame(natural).to_csv(EXPORTS / 'natural_missingness.csv', index=False)
    qc = pd.read_csv(EXPORTS / 'qc_summary_metadata.csv', dtype={'patient_id': str})
    patient_manifest = pd.read_csv(EXPORTS / 'ispy2_patient_manifest.csv', dtype={'patient_id': str})
    missing = pd.DataFrame([dict(patient_id=row.patient_id, timepoint=timepoint, study_uid=None,
                                 qc_status='missing', exclusion_reason='missing_visit', QC_flags='missing_visit')
                            for row in patient_manifest.itertuples() for timepoint in TP2
                            if not bool(getattr(row, f'has_{timepoint}'))])
    pd.concat([qc, missing], ignore_index=True).to_csv(EXPORTS / 'qc_summary.csv', index=False)
    print(f'wrote {len(versions)} versions, {len(fidelity)} fidelity rows, {len(efficiency)} efficiency rows')


if __name__ == '__main__':
    main()
