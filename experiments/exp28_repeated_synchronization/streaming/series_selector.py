"""Select exactly the SEG, early-PE and cropped-DCE series needed for Tier 2."""
import re


def candidates(series, patient_id, study_uid):
    patient_dicom_id = f'ISPY2-{patient_id}'
    rows = series[(series.PatientID == patient_dicom_id) & (series.StudyInstanceUID == study_uid)]
    seg = rows[rows.kind == 'analysis_mask_seg']
    dce = rows[rows.kind == 'cropped_dce']
    pe = rows[rows.kind == 'pe_map']
    if len(seg) != 1 or len(dce) != 1 or len(pe) < 1:
        raise ValueError(f'series selection failed for {patient_id} {study_uid}: SEG={len(seg)} DCE={len(dce)} PE={len(pe)}')
    return seg.iloc[0], dce.iloc[0], pe


def early_pe(pe_rows, phase_indices):
    if len(phase_indices) < 2:
        raise ValueError(f'SEG lacks early/late DCE phase indices: {phase_indices}')
    early = int(phase_indices[1])
    matched = pe_rows[pe_rows.SeriesDescription.str.extract(r': PE(\d+)\s*$', expand=False).astype(float) == early]
    if len(matched) != 1:
        raise ValueError(f'expected one PE{early} series, found {len(matched)}')
    return matched.iloc[0], early


def record(row):
    return dict(series_uid=row.SeriesInstanceUID, series_description=row.SeriesDescription,
                size_bytes=int(round(float(row.series_size_MB) * 2**20)), aws_url=row.series_aws_url)
