"""Read-only DICOM index of the verified HCC-TACE-Seg download.

series_index.csv : one row per series (677) with the header fields needed for pre/post-TACE series selection.
seg_index.csv    : one row per SEG segment (label, referenced CT series, voxel count, 3D connected components).
Nothing is written to the source data. Series directories come from the download verification report.
"""
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage

VERIFY = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'HCC-TACE-Seg/verify_report.tsv'))
OUT = Path(sys.argv[1])

CT_TAGS = ['PatientID', 'StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyDescription', 'SeriesInstanceUID', 'SeriesNumber',
           'SeriesDescription', 'SeriesDate', 'SeriesTime', 'AcquisitionTime', 'Modality', 'SliceThickness', 'ContrastBolusAgent',
           'ContrastBolusRoute', 'ContrastBolusStartTime', 'ConvolutionKernel', 'ProtocolName', 'BodyPartExamined', 'Manufacturer',
           'ManufacturerModelName', 'Rows', 'Columns', 'KVP']


def _s(v):
    if v is None:
        return None
    if isinstance(v, pydicom.multival.MultiValue):
        return '\\'.join(str(x) for x in v)
    return str(v)


def read_series(series_dir):
    files = sorted(Path(series_dir).glob('*.dcm'))
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    rec = {t: _s(ds.get(t)) for t in CT_TAGS}
    rec.update(ImageType=_s(ds.get('ImageType')), PixelSpacing=_s(ds.get('PixelSpacing')), n_files=len(files), series_dir=series_dir)
    if ds.Modality == 'CT':
        z = sorted(float(pydicom.dcmread(f, stop_before_pixels=True, specific_tags=['ImagePositionPatient']).ImagePositionPatient[2])
                   for f in files)
        dz = np.diff(z)
        rec.update(z_extent_mm=z[-1] - z[0] if len(z) > 1 else 0.0, median_z_spacing_mm=float(np.median(dz)) if len(dz) else None,
                   z_spacing_irregular=bool(len(dz) and (np.ptp(dz) > 0.1)))
    return rec


def read_seg(series_dir):
    ds = pydicom.dcmread(next(Path(series_dir).glob('*.dcm')))
    refs = [s.SeriesInstanceUID for s in ds.get('ReferencedSeriesSequence', [])]
    n_ref_inst = sum(len(s.get('ReferencedInstanceSequence', [])) for s in ds.get('ReferencedSeriesSequence', []))
    px = ds.pixel_array
    px = px[None] if px.ndim == 2 else px
    frames = {}
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        seg_no = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        z = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
        frames.setdefault(seg_no, []).append((z, i))
    sp = [float(x) for x in ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing]
    thk = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].get('SliceThickness')
    out = []
    for seg in ds.SegmentSequence:
        n = int(seg.SegmentNumber)
        fl = sorted(frames.get(n, []))
        vol = np.stack([px[i] > 0 for _, i in fl]) if fl else np.zeros((0, 1, 1), bool)
        lab, ncomp = ndimage.label(vol) if vol.size else (None, 0)
        comp_sizes = sorted(np.bincount(lab.ravel())[1:].tolist(), reverse=True) if ncomp else []
        zs = [z for z, _ in fl]
        out.append(dict(
            PatientID=ds.PatientID, seg_series_uid=ds.SeriesInstanceUID, seg_study_uid=ds.StudyInstanceUID, seg_study_date=ds.get('StudyDate'),
            seg_series_description=ds.get('SeriesDescription'), content_creator=_s(ds.get('ContentCreatorName')),
            referenced_series_uids=';'.join(refs), n_referenced_instances=n_ref_inst,
            segment_number=n, segment_label=seg.get('SegmentLabel'),
            category=seg.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning if 'SegmentedPropertyCategoryCodeSequence' in seg else None,
            property_type=seg.SegmentedPropertyTypeCodeSequence[0].CodeMeaning if 'SegmentedPropertyTypeCodeSequence' in seg else None,
            algorithm_type=seg.get('SegmentAlgorithmType'), n_frames=len(fl), voxels=int(vol.sum()),
            voxel_volume_mm3=sp[0] * sp[1] * (float(thk) if thk else float('nan')), z_min=min(zs) if zs else None, z_max=max(zs) if zs else None,
            n_components_3d=int(ncomp), component_voxels=json.dumps(comp_sizes[:20])))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    v = pd.read_csv(VERIFY, sep='\t')
    assert (v.status == 'OK').all() and len(v) == 677, 'download verification report is not all-OK'
    with ProcessPoolExecutor(32) as ex:
        series = list(ex.map(read_series, v.dirs, chunksize=4))
        segs = [r for rs in ex.map(read_seg, v[v.Modality == "SEG"].dirs) for r in rs]
    s = pd.DataFrame(series)
    assert set(s.SeriesInstanceUID) == set(v.SeriesInstanceUID)
    s.to_csv(OUT / 'series_index.csv', index=False)
    pd.DataFrame(segs).to_csv(OUT / 'seg_index.csv', index=False)
    print(f'{len(s)} series, {len(segs)} SEG segments -> {OUT}')


if __name__ == '__main__':
    main()
