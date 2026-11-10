"""Controlled intermediate-visit hiding and explicit late-arrival restoration (EXPERIMENTAL STRESS TEST).

The hidden visit is real data withheld on purpose; it is recorded as `hidden_stress_test`, never as naturally missing.
run_one() performs the test; verify() recomputes every metric from the stored versions (used on resume, and by tests).
"""
from evaluation.continuity import state_checks
from state.synchronization import observation_paths, synchronize


def run_one(source, store, patient_id, cfg, hidden_timepoint='T1'):
    state = None
    for timepoint in cfg['timepoints']:
        clinical = source.clinical(patient_id) if state is None else None
        observation = source.observation(patient_id, timepoint)
        if timepoint == hidden_timepoint:
            observation = {'status': 'hidden_stress_test', 'reason': 'artificially hidden for Experiment 2.8 stress test'}
        state, event = synchronize(state, timepoint, observation, clinical=clinical, cfg=cfg, now=f'stress-scheduled-{timepoint}')
        store.write(state, event)
    restored = source.observation(patient_id, hidden_timepoint)
    state, event = synchronize(state, hidden_timepoint, restored, cfg=cfg, event_type='late_arrival', now='stress-late-arrival')
    store.write(state, event)
    return verify(store, patient_id, hidden_timepoint)


def verify(store, patient_id, hidden_timepoint='T1'):
    versions = store.all_versions(patient_id)
    scheduled, final = versions[:-1], versions[-1]
    before = scheduled[-1]
    hidden_from = next(i for i, s in enumerate(scheduled) if s['current_timepoint'] == hidden_timepoint)
    while_hidden = scheduled[hidden_from:]
    before_paths, after_paths = observation_paths(before), observation_paths(final)
    prefix = f'history.{hidden_timepoint}.'
    unrelated_preserved = all(after_paths.get(k) == v for k, v in before_paths.items() if not k.startswith(prefix))
    checks = state_checks(final, before)
    lineage = store.lineage(patient_id)['versions']
    return dict(patient_id=str(patient_id), hidden_timepoint=hidden_timepoint,
                missing_recorded_correctly=all(s['observation_status'].get(hidden_timepoint) == 'hidden_stress_test' for s in while_hidden),
                silent_imputation=any(hidden_timepoint in s['history'] for s in while_hidden),
                late_update_success=(final['observation_status'].get(hidden_timepoint) == 'late_arrival'
                                     and lineage[-1]['event_type'] == 'late_arrival'
                                     and final['history'][hidden_timepoint]['fields'].get('ftv_pe_cc', {}).get('value') is not None),
                history_preserved=unrelated_preserved and not checks['historical_mutation_detected'],
                lineage_integrity=store.verify(patient_id), late_event_id=final['sync_event_id'], final_state_version=final['state_version'])

