"""Build the audited visit / patient manifests, cohort flow and visit-level QC for I-SPY2 (no imaging pixels read).

Sources: IDC index (series metadata), official FTV from SEG headers (scripts/harvest_seg_headers.py),
TCIA clinical spreadsheet, UCSF multi-feature measurement table (LD etc., 4-visit patients only).
"""
import json

import numpy as np
import pandas as pd

from ingestion.clinical_loader import ispy2_clinical, ispy2_measurements
from ingestion.metadata_loader import derived_kind, load_series
from ingestion.visit_mapper import ORDER, map_ispy2_visits
from paths import EXPORTS, metadata_dir

FTV_MAX_PLAUSIBLE_CC = 1500.0   # whole-breast scale; larger values are implausible for a tumour FTV


def load_headers():
    h = pd.DataFrame([json.loads(l) for l in (metadata_dir() / 'ispy2_seg_headers.jsonl').read_text().splitlines()])
    return h.drop_duplicates('seg_series_uid_index', keep='last')


def build():
    series, meta = load_series('ispy2')
    k = series.SeriesDescription.map(derived_kind)
    series = series.assign(kind=[a for a, _ in k], laterality_kind=[b for _, b in k])
    st = map_ispy2_visits(series)
    clin, clin_src = ispy2_clinical()
    meas, meas_src = ispy2_measurements()
    hdr = load_headers()
    hdr = hdr[hdr.error == ''].set_index('study_uid_index')
    per = series.groupby(['StudyInstanceUID', 'kind']).agg(n=('SeriesInstanceUID', 'size'), MB=('series_size_MB', 'sum')).unstack(fill_value=0)

    first_date = st.groupby('patient_id').StudyDate.min()  # absolute (shifted) dates are not exported; offsets only
    rows = []
    for r in st.itertuples():
        g = lambda kind, col='n': float(per.loc[r.StudyInstanceUID, (col, kind)]) if (col, kind) in per.columns else 0.0
        h = hdr.loc[r.StudyInstanceUID] if r.StudyInstanceUID in hdr.index else None
        flags = [f for f in r.qc_flags.split(';') if f]
        meas_avail = []
        if h is not None and pd.notna(h.ftv_pe_cc):
            meas_avail += ['FTVpe_official', 'FTVser_official']
            if 'PEearly>70%' not in str(h.ftv_pe_label):
                flags.append('nonstandard_PE_threshold')
            if h.clinical_trial_timepoint and h.clinical_trial_timepoint != r.timepoint:
                flags.append('timepoint_tag_disagrees')
            if not (0 <= h.ftv_pe_cc <= FTV_MAX_PLAUSIBLE_CC):
                flags.append('implausible_ftv')
            if h.ftv_pe_cc == 0:
                flags.append('ftv_zero')  # informational: no enhancing tumour above threshold
        else:
            flags.append('missing_ftv')
        m = meas[meas.patient_id == r.patient_id]
        if len(m) and r.timepoint in ORDER:
            meas_avail += ['LD_table', 'sphericity_table', 'BPE_table', 'tumour_volume_table']
        c = clin[clin.patient_id == r.patient_id]
        rows.append(dict(
            patient_id=r.patient_id, timepoint=r.timepoint, source_study_label=r.StudyDescription, study_uid=r.StudyInstanceUID,
            day_offset=int((r.StudyDate - first_date[r.patient_id]).days), n_series=r.n_series,
            has_DCE=bool(g('cropped_dce') + g('derived_dce')), has_cropped_DCE=bool(g('cropped_dce')), has_FTV_mask=bool(g('analysis_mask_seg')),
            has_SEG=bool(g('analysis_mask_seg')), has_PE_map=bool(g('pe_map')), has_SER_map=bool(g('ser_map')),
            mask_laterality='bilateral' if g('analysis_mask_seg') and per.loc[r.StudyInstanceUID].get(('n', 'cropped_dce'), 0) == 0 else 'unilateral_cropped',
            ftv_pe_cc_official=None if h is None else h.ftv_pe_cc, ftv_ser_cc_official=None if h is None else h.ftv_ser_cc,
            ftv_label=None if h is None else h.ftv_pe_label, seg_series_uid=None if h is None else h.seg_series_uid,
            available_measurements=';'.join(meas_avail),
            bytes_seg=g('analysis_mask_seg', 'MB') * 2**20, bytes_cropped_dce=g('cropped_dce', 'MB') * 2**20, bytes_derived_dce=g('derived_dce', 'MB') * 2**20,
            bytes_pe=g('pe_map', 'MB') * 2**20, bytes_ser=g('ser_map', 'MB') * 2**20, bytes_acquired=g('acquired', 'MB') * 2**20,
            bytes_header_fetched=None if h is None else h.bytes_fetched, estimated_download_bytes=r.study_size_MB * 2**20,
            treatment_arm=c.Arm.iloc[0] if len(c) else None, HR=c.HR.iloc[0] if len(c) else None, HER2=c.HER2.iloc[0] if len(c) else None,
            clinical_subtype=subtype(c), QC_flags=';'.join(flags)))
    vm = pd.DataFrame(rows).sort_values(['patient_id', 'timepoint'])

    BLOCKING = {'ambiguous_timepoint', 'conflicting_patient_id', 'duplicate_study_for_timepoint', 'unexpected_time_ordering',
                'missing_ftv', 'implausible_ftv', 'timepoint_tag_disagrees'}
    vm['valid_visit'] = vm.QC_flags.map(lambda s: not (set(s.split(';')) & BLOCKING))
    pm = []
    for p, g_ in vm.groupby('patient_id'):
        v = set(g_[g_.valid_visit].timepoint)
        c = clin[clin.patient_id == p]
        pm.append(dict(patient_id=p, **{f'has_{t}': t in set(g_.timepoint) for t in ORDER}, **{f'valid_{t}': t in v for t in ORDER},
                       n_visits=g_.timepoint.nunique(), n_valid_visits=len(v),
                       complete_T0_T1_T2={'T0', 'T1', 'T2'} <= v, complete_T0_T1_T2_T3=set(ORDER) <= v,
                       available_FTV=int(g_.ftv_pe_cc_official.notna().sum()), available_diameter=int((meas.patient_id == p).any()) * 4,
                       available_mask=int(g_.has_FTV_mask.sum()), in_measurement_table=bool((meas.patient_id == p).any()),
                       treatment_arm=c.Arm.iloc[0] if len(c) else None, HR=c.HR.iloc[0] if len(c) else None, HER2=c.HER2.iloc[0] if len(c) else None,
                       clinical_subtype=subtype(c), outcome_available=bool(len(c) and pd.notna(c.pCR.iloc[0])),
                       visit_pattern=''.join(t[1] if t in v else '-' for t in ORDER)))
    pm = pd.DataFrame(pm)
    flow = [('patients in IDC ispy2 collection', series.PatientID.nunique()),
            ('patients with clinical record', int(pm.treatment_arm.notna().sum())),
            ('>= 1 valid visit', int((pm.n_valid_visits >= 1).sum())),
            ('>= 2 valid visits', int((pm.n_valid_visits >= 2).sum())),
            ('>= 3 valid visits', int((pm.n_valid_visits >= 3).sum())),
            ('all four valid visits (T0-T3)', int(pm.complete_T0_T1_T2_T3.sum())),
            ('T0-T1-T2 complete', int(pm.complete_T0_T1_T2.sum())),
            ('in UCSF multi-feature table (LD etc.)', int(pm.in_measurement_table.sum()))]
    qc = vm.assign(qc_status=np.where(vm.valid_visit, np.where(vm.QC_flags == '', 'pass', 'pass_with_flags'), 'excluded'),
                   exclusion_reason=vm.QC_flags.map(lambda s: ';'.join(sorted(set(s.split(';')) & BLOCKING))))[
        ['patient_id', 'timepoint', 'study_uid', 'qc_status', 'exclusion_reason', 'QC_flags']]
    EXPORTS.mkdir(exist_ok=True)
    vm.to_csv(EXPORTS / 'ispy2_visit_manifest.csv', index=False)
    pm.to_csv(EXPORTS / 'ispy2_patient_manifest.csv', index=False)
    pd.DataFrame(flow, columns=['patient_count_category', 'n']).to_csv(EXPORTS / 'cohort_flow.csv', index=False)
    qc.to_csv(EXPORTS / 'qc_summary_metadata.csv', index=False)
    (EXPORTS / 'audit_sources.json').write_text(json.dumps(dict(idc=meta, clinical=clin_src, measurements=meas_src,
                                                               n_seg_headers=len(hdr)), indent=1, default=str))
    return vm, pm, flow, clin, meas


def subtype(c):
    if not len(c):
        return None
    hr, her2 = c.HR.iloc[0], c.HER2.iloc[0]
    return {(1, 0): 'HR+/HER2-', (1, 1): 'HR+/HER2+', (0, 1): 'HR-/HER2+', (0, 0): 'HR-/HER2- (TN)'}.get((hr, her2))
