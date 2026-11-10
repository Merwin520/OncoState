"""Positive controls for the continuity / missingness / leakage / provenance checks.

A zero mutation rate is only meaningful if the check would have caught a mutation. Here every check is run on
deliberately corrupted copies of real I-SPY2 state versions; each corruption must be detected, and the matching
legitimate operation (explicit revision, explicit missing record) must not be flagged.
"""
import copy

from evaluation.continuity import state_checks
from state.synchronization import revise


def _first_value_path(state):
    for tp, rec in state['history'].items():
        for k, f in rec.get('fields', {}).items():
            if isinstance(f.get('value'), (int, float)) and not isinstance(f.get('value'), bool):
                return tp, k
    return None, None


def validate_patient(versions, schedule):
    """versions: real stored versions of one patient (oldest first). Returns one row of detection outcomes."""
    out = {}
    parent, child = versions[-2], versions[-1]
    tp, k = _first_value_path(parent)
    if tp is not None:
        bad = copy.deepcopy(child)
        bad['history'][tp]['fields'][k]['value'] = bad['history'][tp]['fields'][k]['value'] + 1.0     # silent overwrite
        out['silent_mutation_detected'] = state_checks(bad, parent)['historical_mutation_detected']
        good, _ = revise(child, tp, k, child['history'][tp]['fields'][k]['value'] + 1.0, reason='positive control',
                         actor='detector-validation', source='positive control')
        out['explicit_revision_not_flagged'] = not state_checks(good, child)['historical_mutation_detected']
        nop = copy.deepcopy(child)
        nop['history'][tp]['fields'][k]['provenance'].pop('source_file', None)
        out['missing_provenance_detected'] = not state_checks(nop, parent)['provenance_complete']
    absent = [t for t in schedule if child['observation_status'].get(t) == 'not_observed']
    observed = [t for t in schedule if child['observation_status'].get(t) in ('observed', 'late_arrival')]
    if absent and observed:
        locf = copy.deepcopy(child)                       # naive last-observation-carried-forward "synchronizer"
        src = max((t for t in observed if schedule.index(t) < schedule.index(absent[0])), default=observed[0], key=schedule.index)
        locf['history'][absent[0]] = copy.deepcopy(locf['history'][src])
        out['silent_imputation_detected'] = absent[0] in locf['history'] and locf['observation_status'][absent[0]] == 'not_observed'
    early = versions[0]
    later = [t for t in schedule if schedule.index(t) > schedule.index(early['current_timepoint']) and t in child['history']]
    if later:
        leak = copy.deepcopy(early)
        leak['history'][later[0]] = copy.deepcopy(child['history'][later[0]])
        out['future_leak_detected'] = state_checks(leak)['future_visit_visible']
    return out
