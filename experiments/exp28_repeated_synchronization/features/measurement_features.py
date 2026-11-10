"""Audited tabular evidence sources shared by persistent and stateless runs."""
import pandas as pd

from ingestion.clinical_loader import ispy1_clinical, ispy2_clinical, ispy2_measurements
from paths import EXPORTS, config
from state.provenance import field, prov

TP2 = ('T0', 'T1', 'T2', 'T3')
TP1 = ('V0', 'V1', 'V2', 'V3')


def _clean(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, 'item') else value


class ISPY2EvidenceSource:
    """Read official source records through a counted interface."""

    dataset = 'I-SPY2'

    def __init__(self):
        self.visits = pd.read_csv(EXPORTS / 'ispy2_visit_manifest.csv', dtype={'patient_id': str})
        clinical, self.clinical_src = ispy2_clinical()
        measurements, self.measurement_src = ispy2_measurements()
        clinical['patient_id'] = clinical.patient_id.astype(str)
        measurements['patient_id'] = measurements.patient_id.astype(str)
        self._vis = self.visits.set_index(['patient_id', 'timepoint'])
        self._clin = clinical.set_index('patient_id')
        self._meas = measurements.set_index('patient_id')
        self.counter = None

    def _count(self, source_reads=0, series_reads=0, tool_calls=0):
        if self.counter is not None:
            self.counter.source_reads += source_reads
            self.counter.series_reads += series_reads
            self.counter.tool_calls += tool_calls

    def clinical(self, patient_id):
        self._count(source_reads=1)
        row = self._clin.loc[str(patient_id)]
        values = dict(arm=_clean(row.Arm), HR=_clean(row.HR), HER2=_clean(row.HER2), MP=_clean(row.MP),
                      age_at_screening=_clean(row.Age_at_Screening), race=_clean(row.Race),
                      menopausal_status=_clean(row.menopausal_status), ethnicity=_clean(row.ethnicity))
        values['clinical_subtype'] = {(1, 0): 'HR+/HER2-', (1, 1): 'HR+/HER2+',
                                      (0, 1): 'HR-/HER2+', (0, 0): 'HR-/HER2- (TN)'}.get(
                                          (values['HR'], values['HER2']))
        provenance = {key: prov('official_clinical_spreadsheet', self.clinical_src['file'], field_name=key)
                      for key in values}
        return {'fields': values, 'provenance': provenance}

    def observation(self, patient_id, timepoint):
        self._count(source_reads=1)
        key = (str(patient_id), timepoint)
        if key not in self._vis.index:
            return None
        row = self._vis.loc[key]
        self._count(series_reads=1, tool_calls=1)
        header_prov = lambda **extra: prov('official_ftv_seg_header', 'analysis-mask SEG header',
                                           row.study_uid, row.seg_series_uid, **extra)
        fields = {
            'ftv_pe_cc': field(_clean(row.ftv_pe_cc_official), header_prov(private_tag='(0117,10B4)', result='FTVpe'), 'cc'),
            'ftv_ser_cc': field(_clean(row.ftv_ser_cc_official), header_prov(private_tag='(0117,10B4)', result='FTVser'), 'cc'),
            'ftv_threshold_label': field(_clean(row.ftv_label), header_prov(private_tag='(0117,10B5)')),
            'mask_reference': field(_clean(row.seg_series_uid), header_prov(role='analysis_mask')),
        }
        if str(patient_id) in self._meas.index:
            self._count(source_reads=1)
            measurement = self._meas.loc[str(patient_id)]
            index = TP2.index(timepoint)
            table_prov = lambda name: prov('official_trial_measurement_table', self.measurement_src['file'], field_name=name)
            columns = {
                'tumour_volume_table_cc': f'VOLUME_TUM_BLU_V{(index + 1) * 10}',
                'longest_diameter_cm': f'LD_T{index}',
                'sphericity': f'SPHERICITY_T{index}',
                'bpe_5slice_mean': f'BPE_5slice_mean_T{index}',
            }
            for name, column in columns.items():
                unit = 'cc' if name.endswith('_cc') else ('cm' if name.endswith('_cm') else None)
                fields[name] = field(_clean(measurement[column]), table_prov(column), unit)
        flags = [] if pd.isna(row.QC_flags) else [value for value in str(row.QC_flags).split(';') if value]
        return dict(status='observed', study_uid=row.study_uid, series_uids=[row.seg_series_uid],
                    source_timepoint_label=row.source_study_label, study_day_offset=int(row.day_offset),
                    fields=fields, qc_flags=flags)

    def outcome(self, patient_id):
        """Post-surgery outcome remains outside every patient-state version."""
        row = self._clin.loc[str(patient_id)]
        return dict(patient_id=str(patient_id), pCR=_clean(row.pCR))


class ISPY1EvidenceSource:
    """Spreadsheet-only external replication with the unchanged engine."""

    dataset = 'I-SPY1'

    def __init__(self):
        data, self.src = ispy1_clinical()
        data['patient_id'] = data.patient_id.astype(str)
        self.data = data.set_index('patient_id')
        self.visit_map = config('ispy1_optional')['visit_map']
        self.reverse = {value: key for key, value in self.visit_map.items()}
        self.counter = None

    def _count(self, source_reads=0, series_reads=0, tool_calls=0):
        if self.counter is not None:
            self.counter.source_reads += source_reads
            self.counter.series_reads += series_reads
            self.counter.tool_calls += tool_calls

    def clinical(self, patient_id):
        self._count(source_reads=1)
        row = self.data.loc[str(patient_id)]
        columns = ['age', 'race_id', 'ERpos', 'PgRpos', 'HR Pos', 'Her2MostPos',
                   'HR_HER2_CATEGORY', 'HR_HER2_STATUS', 'BilateralCa', 'Laterality']
        values = {column: _clean(row[column]) for column in columns}
        provenance = {column: prov('official_clinical_spreadsheet', self.src['file'], field_name=column)
                      for column in columns}
        return {'fields': values, 'provenance': provenance}

    def observation(self, patient_id, timepoint):
        self._count(source_reads=1)
        row = self.data.loc[str(patient_id)]
        source_name = self.reverse[timepoint]
        value = _clean(row[source_name])
        if value is None:
            return None
        self._count(tool_calls=1)
        provenance = prov('official_clinical_spreadsheet', self.src['file'], field_name=source_name)
        return dict(status='observed', study_uid=None, series_uids=[], source_timepoint_label=source_name,
                    fields={'longest_diameter_mm': field(value, provenance, 'mm')}, qc_flags=[])

    def outcome(self, patient_id):
        row = self.data.loc[str(patient_id)]
        return {key: _clean(row[key]) for key in ['PCR', 'RCBClass', 'RFS', 'rfs_ind', 'sstat', 'survDtD2 (tx)']}


def ispy2_cfg(patient_id):
    return dict(patient_id=str(patient_id), timepoints=list(TP2),
                phase_names=config('ispy2')['timepoint_meaning'],
                measure_keys=['ftv_pe_cc', 'ftv_ser_cc', 'tumour_volume_table_cc',
                              'longest_diameter_cm', 'sphericity', 'bpe_5slice_mean'])


def ispy1_cfg(patient_id):
    return dict(patient_id=str(patient_id), timepoints=list(TP1),
                phase_names={value: key for key, value in config('ispy1_optional')['visit_map'].items()},
                measure_keys=['longest_diameter_mm'])
