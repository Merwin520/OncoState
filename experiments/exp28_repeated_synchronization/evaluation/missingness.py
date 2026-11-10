"""Missingness endpoint helpers."""


def patient_accuracy(table):
    return table.groupby('patient_id').agg(missingness_detection_accuracy=('missing_recorded_correctly', 'mean'),
                                           silent_imputation_rate=('silent_imputation', 'mean')).reset_index()
