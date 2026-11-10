"""MU-Glioma-Post dataset adapter for the multi-visit longitudinal experiment (shared by ALL three systems).

Source (TCIA, CC BY 4.0, doi:10.7937/7K9K-3C83), downloaded unmodified to $MU_GLIOMA_ROOT (default $ONCOSTATE_DATA/MU-Glioma-Post):
  nifti/PKG - MU-Glioma-Post/MU-Glioma-Post/PatientID_XXXX/Timepoint_k/PatientID_XXXX_Timepoint_k_{brain_t1c,brain_t1n,brain_t2f,brain_t2w,tumorMask}.nii.gz
  raw/MU-Glioma-Post_ClinicalData-July2025.xlsx  (converted once to raw/clinical.csv by freeze_sources.py, sha256 recorded)

One frozen measurement function (measure_mask) turns the released tumorMask of a visit into volumes; the reference builder and every
system read visit evidence ONLY through visit_record(), so all systems receive byte-identical evidence. A VisitClock decides which visits
exist "now"; asking for any other visit raises FutureAccess (and is logged), which is the run-time leakage guard.
"""
import hashlib
import json
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('MU_GLIOMA_ROOT', os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'MU-Glioma-Post')))
NIFTI = ROOT / 'nifti/PKG - MU-Glioma-Post/MU-Glioma-Post'
CLINICAL_CSV = ROOT / 'raw/clinical.csv'
MODALITIES = ('brain_t1c', 'brain_t1n', 'brain_t2f', 'brain_t2w')
LABELS = {1: 'netc_ml', 2: 'snfh_ml', 3: 'et_ml', 4: 'resection_cavity_ml'}   # TCIA label definitions (dataset description)
BURDEN_LABELS = (1, 2, 3)                                                     # whole tumour excluding the resection cavity
MEASUREMENT_VERSION = 'mu-glioma-measure-1.0'

# Treatment events with a start day in the same frame as the MRI days (days from diagnosis). Column names after .strip().
TREATMENT_COLUMNS = {
    'surgery': 'Number of days from Diagnosis to First surgery or procedure',
    'radiation': 'Number of days from Diagnosis to Radiation Therapy Start date',
    'initial_chemotherapy': 'Number of days from Diagnosis to Initial Chemo Therapy Start date',
    'additional_therapy': 'Number of Days from Diagnosis to Starting Additional Therapy',
    'second_additional_therapy': 'Number of Days from Diagnosis to Starting 2nd_Additional Therapy',
    'immunotherapy': 'Number of Days from Diagnosis to Start Immunotherapy',
    'brachytherapy': 'Number of Days from Diagnosis to the day of Insertion of Brachytherapy',
    'other_therapy': 'Number of Days from Diagnosis to Start Other Additional Therapy',
}
MRI_DAY_COLUMN = 'Number of Days from Diagnosis to {n} MRI (Timepoint_{k})'
ORDINAL = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 5: '5th', 6: '6th'}


class FutureAccess(RuntimeError):
    """A system asked for a visit that has not (yet) arrived."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tp_index(tp):
    return int(tp.split('_')[1])


def parse_day(v):
    """Frozen rule: numeric -> int; 'a, b' (several starts) -> earliest integer; anything else -> None (unknown)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v)
    nums = [int(x) for x in re.findall(r'-?\d+', str(v))]
    return min(nums) if nums and re.fullmatch(r'\s*-?\d+(\s*,\s*-?\d+)*\s*', str(v)) else None


@lru_cache(maxsize=1)
def clinical():
    c = pd.read_csv(CLINICAL_CSV)
    c.columns = [x.strip() for x in c.columns]
    return c.set_index('Patient_ID')


def mri_day(pid, tp):
    k = tp_index(tp)
    col = MRI_DAY_COLUMN.format(n=ORDINAL[k], k=k)
    c = clinical()
    return parse_day(c.loc[pid, col]) if pid in c.index and col in c.columns else None


def treatment_events(pid):
    """[(category, start_day)] for every treatment with a parseable start day, plus unparseable entries for the manifest."""
    c = clinical()
    if pid not in c.index:
        return [], []
    ev, bad = [], []
    for cat, col in TREATMENT_COLUMNS.items():
        v = c.loc[pid, col]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        d = parse_day(v)
        (ev if d is not None else bad).append((cat, d if d is not None else str(v)))
    return sorted(ev, key=lambda x: (x[1], x[0])), bad


def released_timepoints(pid):
    d = NIFTI / pid
    return sorted([p.name for p in d.iterdir() if p.is_dir() and re.fullmatch(r'Timepoint_\d+', p.name)], key=tp_index) if d.exists() else []


def mask_path(pid, tp):
    return NIFTI / pid / tp / f'{pid}_{tp}_tumorMask.nii.gz'


def available_modalities(pid, tp):
    return [m for m in MODALITIES if (NIFTI / pid / tp / f'{pid}_{tp}_{m}.nii.gz').exists()]


def measure_mask(path):
    """THE frozen reference-processing pipeline: label voxel counts x voxel volume -> mL (2 decimals). Returns (dict, seconds)."""
    import nibabel as nib
    t0 = time.perf_counter()
    img = nib.load(str(path))
    a = np.asarray(img.dataobj).astype(np.int16)
    vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    labels = set(np.unique(a).tolist())
    if not labels <= {0, 1, 2, 3, 4}:
        raise ValueError(f'unexpected labels {sorted(labels)}')
    counts = {k: int((a == k).sum()) for k in LABELS}
    out = {LABELS[k]: round(counts[k] * vox_ml, 2) for k in LABELS}
    out['burden_ml'] = round(sum(counts[k] for k in BURDEN_LABELS) * vox_ml, 2)
    out['voxel_volume_ml'] = vox_ml
    out['shape'] = list(a.shape)
    return out, time.perf_counter() - t0


_measure_cache, _mlock = {}, threading.Lock()


def measurement(pid, tp):
    """Computational cache of measure_mask (NOT clinical state): (values, uncached_compute_seconds)."""
    key = (pid, tp)
    with _mlock:
        if key in _measure_cache:
            return _measure_cache[key]
    val = measure_mask(mask_path(pid, tp))
    with _mlock:
        _measure_cache[key] = val
    return val


def load_measure_cache(path):
    if Path(path).exists():
        _measure_cache.update({tuple(k.split('|')): (v['values'], v['seconds']) for k, v in json.loads(Path(path).read_text()).items()})


def save_measure_cache(path):
    Path(path).write_text(json.dumps({f'{p}|{t}': {'values': v, 'seconds': s} for (p, t), (v, s) in sorted(_measure_cache.items())}, indent=0))


def interval_treatments(pid, visits, tp):
    """Treatment events reported WITH visit tp: start day in (day of latest earlier visit with known day, day of tp].
    None when the visit day is unknown (not evaluable). visits = the chronologically ordered visits of the patient."""
    d = mri_day(pid, tp)
    if d is None:
        return None
    earlier = [mri_day(pid, v) for v in visits[:visits.index(tp)]]
    earlier = [x for x in earlier if x is not None]
    lo = max(earlier) if earlier else None
    ev, _ = treatment_events(pid)
    return [dict(treatment=c, start_day=s) for c, s in ev if s <= d and (lo is None or s > lo)]


def visit_record(pid, tp, visits):
    """The visit evidence every system receives (identical content). visits: the patient's included visits in chronological order."""
    vals, secs = measurement(pid, tp)
    rec = dict(patient_id=pid, visit=tp, days_from_diagnosis=mri_day(pid, tp),
               measurements={k: vals[k] for k in ('burden_ml', 'et_ml', 'netc_ml', 'snfh_ml', 'resection_cavity_ml')},
               treatments_reported_since_previous_visit=interval_treatments(pid, visits, tp),
               source=dict(file=str(mask_path(pid, tp).relative_to(ROOT)), measurement_version=MEASUREMENT_VERSION))
    return rec, secs


class VisitClock:
    """Which visits of a patient exist at the current moment of the simulated timeline (thread-local per patient sequence)."""
    _local = threading.local()

    @classmethod
    def set(cls, pid, visits_in_order, available, current, log=None):
        cls._local.ctx = dict(pid=pid, visits=list(visits_in_order), available=list(available), current=current,
                              log=log if log is not None else [])

    @classmethod
    def get(cls):
        return getattr(cls._local, 'ctx', None)

    @classmethod
    def check(cls, pid, tp):
        ctx = cls.get()
        if ctx is None or ctx['pid'] != pid:
            raise FutureAccess(f'no active timeline for patient {pid}')
        ok = tp in ctx['available']
        ctx['log'].append(dict(pid=pid, visit=tp, current=ctx['current'], allowed=ok, t=time.time()))
        if not ok:
            raise FutureAccess(f'visit {tp} is not available at the current time')
        return ctx

    @classmethod
    def available_visits(cls, pid):
        ctx = cls.get()
        if ctx is None or ctx['pid'] != pid:
            raise FutureAccess(f'no active timeline for patient {pid}')
        return [v for v in ctx['visits'] if v in ctx['available']]       # chronological order
