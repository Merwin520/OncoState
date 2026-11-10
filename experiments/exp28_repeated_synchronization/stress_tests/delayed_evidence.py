"""Secondary out-of-order test: T0 -> T2 -> delayed T1 -> T3 (EXPERIMENTAL; not a central claim).

run_one() performs the sequence; verify() recomputes every metric from the stored versions (used on resume and by tests).
"""
from state.synchronization import observation_paths, synchronize

SEQUENCE = [('T0', 'scheduled'), ('T2', 'scheduled'), ('T1', 'late_arrival'), ('T3', 'scheduled')]


def run_one(source, store, patient_id, cfg):
    state = None
    for timepoint, event_type in SEQUENCE:
        state, event = synchronize(state, timepoint, source.observation(patient_id, timepoint),
                                   clinical=source.clinical(patient_id) if state is None else None,
                                   cfg=cfg, event_type=event_type, now=f'delayed-{timepoint}')
        store.write(state, event)
    return verify(store, patient_id, cfg)


def verify(store, patient_id, cfg):
    versions = store.all_versions(patient_id)
    lineage = store.lineage(patient_id)['versions']
    preserved = True
    for before, after in zip(versions, versions[1:]):
        old, new = observation_paths(before), observation_paths(after)
        preserved &= all(k in new and new[k] == v for k, v in old.items())   # nothing earlier changes when T1 is inserted late
    final = versions[-1]
    return dict(patient_id=str(patient_id), sequence='T0,T2,T1(late),T3',
                delayed_logged_explicitly=[v['event_type'] for v in lineage] == [e for _, e in SEQUENCE],
                all_visits_present=set(final['history']) == set(cfg['timepoints']),
                prior_observations_preserved=bool(preserved),
                late_record_marked=final['observation_status'].get('T1') == 'late_arrival',
                lineage_integrity=store.verify(patient_id), final_state_version=final['state_version'])
