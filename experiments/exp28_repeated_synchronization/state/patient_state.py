"""PatientState construction and derived views. The state is a plain JSON-serialisable dict.

Sections
  history            {timepoint: observation record}  — append-only; changes only through explicit revisions
  observation_status {timepoint: observed | not_observed | hidden_stress_test | late_arrival}
  missingness        visit- and field-level absence with reasons (never imputed)
  tumour_state       DERIVED from history at every version (current value, trajectory, deltas)
  treatment_context, clinical_context   from the clinical source (outcomes such as pCR are excluded)
  revisions, provenance, events          append-only logs
"""
import copy

from state.schemas import FORBIDDEN_STATE_FIELDS, SCHEMA_VERSION


def new_state(patient_id, timepoints, clinical, phase_names):
    bad = set(clinical.get('fields', {})) & set(FORBIDDEN_STATE_FIELDS)
    if bad:
        raise ValueError(f'outcome fields may not enter the state: {bad}')
    return dict(patient_id=str(patient_id), schema_version=SCHEMA_VERSION, state_version=None, version_index=-1, parent_version=None,
                current_timepoint=None, protocol_timepoints=list(timepoints), source_observation=None, sync_event_id=None,
                history={}, observation_status={}, missingness={'visits': {}, 'fields': {}},
                tumour_state={}, treatment_context=dict(arm=clinical['fields'].get('arm'), phase_at_current=None, phase_names=phase_names),
                clinical_context=copy.deepcopy(clinical['fields']), clinical_provenance=clinical['provenance'],
                QC_flags=[], revisions=[], provenance=[], events=[])


def derive_tumour_state(state, measure_keys):
    """Recompute the derived trajectory from the historical observations only (no imputation of absent visits)."""
    tps = [t for t in state['protocol_timepoints'] if t in state['history'] and state['history'][t]['status'] in ('observed', 'late_arrival')]
    traj = {}
    for k in measure_keys:
        seq = [(t, state['history'][t]['fields'].get(k, {}).get('value')) for t in tps]
        seq = [(t, v) for t, v in seq if v is not None]
        rows = {}
        base = next((v for t, v in seq if t == state['protocol_timepoints'][0]), None)
        prev = None
        for t, v in seq:
            rows[t] = dict(value=v, delta_from_previous=None if prev is None else v - prev[1], previous_timepoint=None if prev is None else prev[0],
                           pct_change_from_first_protocol_visit=None if base in (None, 0) else 100 * (v - base) / base)
            prev = (t, v)
        traj[k] = dict(by_timepoint=rows, current=seq[-1][1] if seq else None, current_as_of=seq[-1][0] if seq else None)
    return dict(derived=True, computed_from_timepoints=tps, measures=traj)
