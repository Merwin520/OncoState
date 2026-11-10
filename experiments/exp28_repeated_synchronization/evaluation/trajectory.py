"""Trajectory endpoint helpers."""
import pandas as pd


def patient_metrics(table):
    data = table.copy()
    numeric = data[data.reference_value.notna() & data.state_value.notna()]
    return numeric.groupby('patient_id').agg(mae=('absolute_error', 'mean'),
                                             relative_error=('relative_error', 'mean'),
                                             n_observations=('timepoint', 'size')).reset_index()


def direction_metrics(table):
    data = table[table.direction_correct.notna()].copy()
    if not len(data):
        return pd.DataFrame(columns=['patient_id', 'direction_accuracy'])
    data['direction_correct'] = data.direction_correct.astype(bool)
    return data.groupby('patient_id').direction_correct.mean().rename('direction_accuracy').reset_index()
