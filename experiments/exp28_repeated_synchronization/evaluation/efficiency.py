"""Paired patient-level efficiency table construction."""


def paired_by_timepoint(table, metric):
    return table.pivot_table(index=['patient_id', 'timepoint'], columns='strategy', values=metric).dropna().reset_index()
