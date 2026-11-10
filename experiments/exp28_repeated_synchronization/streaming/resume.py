"""Atomic processing-status ledger for restart without completed-visit redownload."""
from pathlib import Path

import pandas as pd

COLUMNS = ['patient_id', 'timepoint', 'metadata_audited', 'selected', 'download_started',
           'download_complete', 'features_extracted', 'state_created', 'validated', 'raw_deleted', 'error']


def completed(row):
    """A visit is finished when its extraction outcome (pass or recorded failure) is stored and raw data are gone."""
    return bool(row.features_extracted) and bool(row.raw_deleted)


class StatusLedger:
    def __init__(self, path, rows):
        self.path = Path(path)
        if self.path.exists():
            self.data = pd.read_csv(self.path, dtype={'patient_id': str}).fillna({'error': ''})
        else:
            self.data = pd.DataFrame(rows, columns=COLUMNS)
            self.save()

    def row(self, patient_id, timepoint):
        match = (self.data.patient_id == str(patient_id)) & (self.data.timepoint == timepoint)
        if match.sum() != 1:
            raise KeyError((patient_id, timepoint))
        return self.data.loc[match].iloc[0]

    def update(self, patient_id, timepoint, **values):
        match = (self.data.patient_id == str(patient_id)) & (self.data.timepoint == timepoint)
        for key, value in values.items():
            self.data.loc[match, key] = value
        self.save()

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        self.data.to_csv(temporary, index=False)
        temporary.replace(self.path)
