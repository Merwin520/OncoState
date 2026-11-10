"""State-continuity, invariance, provenance and leakage checks."""
from state.provenance import complete
from state.schemas import FORBIDDEN_STATE_FIELDS
from state.synchronization import observation_paths


def state_checks(state, previous=None):
    history = state.get('history', {})
    fields = [value for record in history.values() for value in record.get('fields', {}).values()]
    provenance_complete = all(complete(value.get('provenance')) for value in fields)
    mutation = False
    retained = 1.0
    if previous is not None:
        before, after = observation_paths(previous), observation_paths(state)
        historical = {key: value for key, value in before.items() if key.startswith('history.')}
        kept = sum(key in after and after[key] == value for key, value in historical.items())
        retained = kept / len(historical) if historical else 1.0
        revised = {f"history.{row['timepoint']}.fields.{row['field']}.value" for row in state.get('revisions', [])}
        mutation = any(key in after and after[key] != value and key not in revised for key, value in historical.items())
    order = state['protocol_timepoints']
    current_index = order.index(state['current_timepoint']) if state.get('current_timepoint') in order else -1
    future_visible = any(order.index(tp) > current_index for tp in history if tp in order)
    forbidden = set()
    stack = [state]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            forbidden |= set(item) & set(FORBIDDEN_STATE_FIELDS)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return dict(history_retention_rate=retained, provenance_complete=provenance_complete,
                historical_mutation_detected=mutation, future_visit_visible=future_visible,
                forbidden_outcome_fields=';'.join(sorted(forbidden)))
