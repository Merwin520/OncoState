"""Patient-by-patient Tier 2 extraction with verification, deletion and resume."""
import json
from pathlib import Path

import pandas as pd
import pydicom

from features.imaging_features import extract as imaging_extract, save_thumbnail
from features.mask_features import extract as mask_extract
from paths import EXPORTS, PKG, STATES, cache_dir
from state.schemas import EXTRACTION_VERSION, PROCESSING_VERSION
from streaming.downloader import SeriesDownloader
from streaming.extract_and_delete import delete_raw
from streaming.idc_metadata import ispy2_series
from streaming.resume import StatusLedger, completed
from streaming.series_selector import candidates, early_pe, record
from streaming.storage_guard import check, used_bytes


def _dicoms(path, with_pixels=()):
    """Read headers only, except for series whose pixel data are needed (keeps memory low; byte sizes were checked at download)."""
    datasets = []
    for file in sorted(Path(path).rglob('*')):
        if not file.is_file():
            continue
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            if str(ds.SeriesInstanceUID) in with_pixels:
                ds = pydicom.dcmread(file)
            datasets.append(ds)
        except Exception as error:
            raise ValueError(f'corrupted download {file.name}: {error}') from error
    return datasets


def _by_series(datasets):
    output = {}
    for dataset in datasets:
        output.setdefault(str(dataset.SeriesInstanceUID), []).append(dataset)
    return output


def _upsert_feature(path, row):
    data = pd.read_csv(path, dtype={'patient_id': str}) if path.exists() else pd.DataFrame()
    if len(data):
        keep = ~((data.patient_id == str(row['patient_id'])) & (data.timepoint == row['timepoint']))
        data = data.loc[keep]
    data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
    temporary = path.with_suffix('.tmp')
    data.sort_values(['patient_id', 'timepoint']).to_csv(temporary, index=False)
    temporary.replace(path)


def representative_patient(visits, selected):
    baseline = visits[(visits.patient_id.isin(selected)) & (visits.timepoint == 'T0')].copy()
    median = baseline.ftv_pe_cc_official.median()
    baseline['distance'] = (baseline.ftv_pe_cc_official - median).abs()
    return baseline.sort_values(['distance', 'patient_id']).iloc[0].patient_id


def process_eligible(max_cache_gb=10, limit=None):
    visits = pd.read_csv(EXPORTS / 'ispy2_visit_manifest.csv', dtype={'patient_id': str})
    selected = pd.read_csv(EXPORTS / 'tier2_subset.csv', dtype={'patient_id': str}).patient_id.tolist()
    selected_rows = visits[visits.patient_id.isin(selected)].sort_values(['patient_id', 'timepoint'])
    status_rows = [dict(patient_id=row.patient_id, timepoint=row.timepoint, metadata_audited=True,
                        selected=True, download_started=False, download_complete=False,
                        features_extracted=False, state_created=(STATES / row.patient_id / 'lineage.json').exists(),
                        validated=False, raw_deleted=False, error='')
                   for row in selected_rows.itertuples()]
    ledger = StatusLedger(EXPORTS / 'processing_status.csv', status_rows)
    series, metadata = ispy2_series()
    downloader = SeriesDownloader(series)
    feature_path = EXPORTS / 'imaging_validation.csv'
    representative = representative_patient(visits, selected)
    (EXPORTS / 'representative_case.json').write_text(json.dumps(
        dict(patient_id=representative, rule='Tier-2 patient nearest the Tier-2 median baseline official FTV; patient ID tie-break; outcome-blind'), indent=1))
    errors = []
    todo = list(selected_rows.itertuples())[:limit]
    for number, row in enumerate(todo, 1):
        current = ledger.row(row.patient_id, row.timepoint)
        if completed(current) and feature_path.exists():
            continue
        visit_dir = cache_dir() / f'{row.patient_id}_{row.timepoint}'
        try:
            seg_row, dce_row, pe_rows = candidates(series, row.patient_id, row.study_uid)
            seg = record(seg_row)
            ledger.update(row.patient_id, row.timepoint, download_started=True, error='')
            check(cache_dir(), seg['size_bytes'], max_cache_gb)
            visit_dir.mkdir(parents=True, exist_ok=True)
            downloader.download([seg['series_uid']], visit_dir)
            datasets = _dicoms(visit_dir, with_pixels={seg['series_uid']})
            grouped = _by_series(datasets)
            if seg['series_uid'] not in grouped or len(grouped[seg['series_uid']]) != 1:
                raise ValueError('downloaded SEG is missing or duplicated')
            mask, header, mask_features = mask_extract(grouped[seg['series_uid']][0])
            pe_row, early_index = early_pe(pe_rows, header['dce_phase_indices'])
            pe, dce = record(pe_row), record(dce_row)
            check(cache_dir(), pe['size_bytes'] + dce['size_bytes'], max_cache_gb)
            downloader.download([pe['series_uid'], dce['series_uid']], visit_dir)
            ledger.update(row.patient_id, row.timepoint, download_complete=True)
            datasets = _dicoms(visit_dir, with_pixels={seg['series_uid'], pe['series_uid']})
            grouped = _by_series(datasets)
            expected = {seg['series_uid'], pe['series_uid'], dce['series_uid']}
            if not expected <= set(grouped):
                raise ValueError(f'missing downloaded series: {expected - set(grouped)}')
            image_features = imaging_extract(mask, grouped[pe['series_uid']], grouped[dce['series_uid']])
            official = float(row.ftv_pe_cc_official)
            error = abs(mask_features['local_ftv_recomputed_cc'] - official)
            header_voxels = header.get('ftv_pe_voxels')
            patient_match = header['patient_dicom_id'] == f'ISPY2-{row.patient_id}'
            study_match = header['study_uid'] == row.study_uid
            series_match = header['seg_series_uid'] == seg['series_uid']
            identity_ok = patient_match and study_match and series_match
            geometry_ok = image_features['pe_geometry_match'] and image_features['dce_geometry_match']
            ftv_ok = error < .01
            valid = identity_ok and geometry_ok and ftv_ok
            reasons = [r for r, bad in (('identity_mismatch', not identity_ok), ('mask_image_geometry_mismatch', not geometry_ok),
                                        (f"local_ftv_differs_from_official (mask FTV voxels {mask_features['local_ftv_voxels']} vs header {header_voxels})",
                                         not ftv_ok)) if bad]
            result = dict(patient_id=row.patient_id, timepoint=row.timepoint, study_uid=row.study_uid,
                          official_ftv_pe_cc=official, absolute_error_cc=error,
                          early_pe_phase_index=early_index, seg_series_uid=seg['series_uid'],
                          pe_series_uid=pe['series_uid'], dce_series_uid=dce['series_uid'],
                          patient_id_match=patient_match, study_uid_match=study_match, series_uid_match=series_match,
                          header_ftv_voxels=header_voxels, voxel_count_difference=None if header_voxels is None else mask_features['local_ftv_voxels'] - header_voxels,
                          validated=valid, failure_reason=';'.join(reasons), source_type='local_ftv_recomputed', source_file='streamed DICOM objects',
                          source_study_uid=row.study_uid, source_series_uid=seg['series_uid'],
                          extraction_version=EXTRACTION_VERSION, processing_version=PROCESSING_VERSION,
                          bytes_selected=seg['size_bytes'] + pe['size_bytes'] + dce['size_bytes'],
                          idc_version=metadata.get('idc_version'), **mask_features, **image_features)
            if row.patient_id == representative:
                asset = PKG / 'figures' / 'assets' / f'representative_{row.timepoint}.png'
                dce_full = _by_series(_dicoms(visit_dir, with_pixels={dce['series_uid']}))[dce['series_uid']]
                save_thumbnail(mask, dce_full, asset)
                result['thumbnail'] = str(asset.relative_to(PKG))
            else:
                result['thumbnail'] = None
            _upsert_feature(feature_path, result)
            ledger.update(row.patient_id, row.timepoint, features_extracted=True, validated=valid, error=';'.join(reasons))
            # the outcome (pass, or the diagnosed failure) is now on record; raw imaging is deleted either way
            delete_raw(visit_dir, validated=valid, diagnostics_recorded=True)
            ledger.update(row.patient_id, row.timepoint, raw_deleted=True)
            print(f"Tier 2 {number}/{len(todo)} {row.patient_id} {row.timepoint} {'validated' if valid else 'FAILED: ' + ';'.join(reasons)}; "
                  f"cache={used_bytes(cache_dir())}", flush=True)
        except Exception as error:  # download/parse failure: recorded, partial files removed, never dropped silently
            ledger.update(row.patient_id, row.timepoint, error=f'processing_error: {error}')
            errors.append((row.patient_id, row.timepoint, str(error)))
            delete_raw(visit_dir, diagnostics_recorded=True)
            ledger.update(row.patient_id, row.timepoint, raw_deleted=True)
            print(f'Tier 2 ERROR {row.patient_id} {row.timepoint}: {error}', flush=True)
    status = ledger.data
    qc = status.assign(qc_status=status.apply(lambda value: 'pass' if value.validated else 'failed', axis=1),
                       exclusion_reason=status.error.fillna(''))[['patient_id', 'timepoint', 'qc_status', 'exclusion_reason']]
    qc.to_csv(EXPORTS / 'qc_imaging.csv', index=False)
    if errors:
        raise RuntimeError(f'{len(errors)} Tier 2 visits hit processing errors; see processing_status.csv')
    if limit is None and not status.apply(completed, axis=1).all():
        raise RuntimeError('Tier 2 did not finish every selected visit')


if __name__ == '__main__':
    process_eligible()
