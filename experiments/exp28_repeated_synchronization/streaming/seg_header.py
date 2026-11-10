"""Header-only extraction of the official UCSF FTV results from I-SPY2 analysis-mask DICOM SEG objects.

The FTV computed by the trial's UCSF pipeline is stored in private tags of the SEG header:
(0117,10B0) FTV Sequence -> (0117,10B4) Volume [cc], (0117,10B3) voxel count, (0117,10B5) label.
Only the first bytes of each object are fetched with an HTTP range request from the public IDC bucket and parsed in
memory, so no imaging data is written to disk (cache use is ~0). Pixel data are never read.
"""
import io
import re
import time
import urllib.request

import pydicom

from paths import config

PRIV = 0x0117
BLOCK = 0x10  # private block reserved by (0117,0010) 'UCSF BIRP PRIVATE CREATOR 011710xx'


def _ours(el):
    return el.tag.group == PRIV and (el.tag.element >> 8) == BLOCK


def _get(url, nbytes=None, timeout=60):
    req = urllib.request.Request(url, headers={'Range': f'bytes=0-{nbytes - 1}'} if nbytes else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def object_url(series_aws_url):
    """s3://bucket/<crdc_series_uuid>/*  ->  https URL of the single instance (SEG objects are single-file)."""
    bucket, prefix = series_aws_url.split('/')[2], series_aws_url.split('/')[3]
    listing = _get(f'https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}/').decode()
    keys = re.findall(r'<Key>([^<]+)</Key>', listing)
    sizes = [int(s) for s in re.findall(r'<Size>(\d+)</Size>', listing)]
    if len(keys) != 1:
        raise ValueError(f'expected one instance, found {len(keys)}')
    return f'https://{bucket}.s3.amazonaws.com/{keys[0]}', sizes[0]


def _val(ds, elem):
    for el in ds:
        if _ours(el) and (el.tag.element & 0xFF) == elem:
            return el.value
    return None


def parse_header(ds):
    ftv = []
    seq = _val(ds, 0xB0)
    for it in (seq or []):
        rec = {}
        for el in it:
            if _ours(el):
                rec[{0xB1: 'ser_min', 0xB2: 'ser_max_code', 0xB3: 'voxels', 0xB4: 'volume_cc', 0xB5: 'label'}.get(el.tag.element & 0xFF, hex(el.tag.element))] = str(el.value)
        ftv.append(rec)
    pe = [r for r in ftv if str(r.get('label', '')).startswith('FTVpe')]
    ser = [r for r in ftv if str(r.get('label', '')).startswith('FTVser')]
    qc = {}
    for el in ds:
        if _ours(el) and 0xC0 <= (el.tag.element & 0xFF) <= 0xC5:
            qc[hex(el.tag.element)] = str(el.value)[:200]
    phase_indices = _val(ds, 0x35)
    voxel_size = _val(ds, 0x94)
    return dict(patient_dicom_id=str(ds.get('PatientID', '')), study_uid=str(ds.get('StudyInstanceUID', '')), seg_series_uid=str(ds.get('SeriesInstanceUID', '')),
                study_date=str(ds.get('StudyDate', '')), clinical_trial_timepoint=str(ds.get('ClinicalTrialTimePointID', '') or ''),
                laterality=str(_val(ds, 0x98) or ''), n_ftv_results=len(ftv),
                ftv_pe_cc=float(pe[0]['volume_cc']) if pe and pe[0].get('volume_cc') not in (None, '') else None,
                ftv_pe_voxels=int(float(pe[0]['voxels'])) if pe and pe[0].get('voxels') else None, ftv_pe_label=pe[0].get('label') if pe else None,
                ftv_ser_cc=float(ser[0]['volume_cc']) if ser and ser[0].get('volume_cc') not in (None, '') else None,
                ftv_ser_label=ser[0].get('label') if ser else None,
                mask_operations=str(_val(ds, 0xD2) or ''), process_descriptions=str(_val(ds, 0xD4) or '')[:400],
                protocol_compliance=str(_val(ds, 0xC5) or ''), qc_private=qc,
                dce_phase_indices=[int(x) for x in phase_indices] if phase_indices is not None else [],
                resampled_voxel_mm=[float(x) for x in voxel_size] if voxel_size is not None else [],
                rows=ds.get('Rows'), columns=ds.get('Columns'), n_frames=ds.get('NumberOfFrames'))


def fetch(series_aws_url, cfg=None):
    """Return parsed header fields plus transfer accounting. Raises on failure (caller records the error)."""
    cfg = cfg or config()
    t0 = time.time()
    url, size = object_url(series_aws_url)
    got = 0
    for n in (cfg['header_fetch_bytes'], cfg['header_fetch_retry_bytes'], size):
        b = _get(url, min(n, size))
        got += len(b)
        try:
            ds = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True, force=True)
            out = parse_header(ds)
            if out['n_ftv_results'] or n >= size:
                return dict(out, object_bytes=size, bytes_fetched=got, fetch_seconds=round(time.time() - t0, 3), source_url=url)
        except Exception:
            if n >= size:
                raise
    raise RuntimeError('header not parsed')
