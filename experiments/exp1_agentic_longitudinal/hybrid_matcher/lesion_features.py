#!/usr/bin/env python3
"""Per-lesion GT-free features for the hybrid matcher: geometry from the Exp 2.6 case files, appearance and shape from the setting-A
lesion masks (cut_A) on the original CT. Writes outputs/hybrid_matcher/lesions.csv (one row per lesion)."""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / 'exp26_clinician_coevolution/data/real/exp22_longitudinal_ct/cases'
MASKS = REPO / 'experiments/exp1_agentic_longitudinal/outputs/exp22_baselines/exp22_regen/cut_A'
CT = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'Longitudinal-CT'))
OUT = REPO / 'outputs/hybrid_matcher'


def one(path):
    c = json.loads(path.read_text()); pid = c['patient_id']; rows = []
    for side, key, exam in (('baseline', 'baseline_lesions', 'baseline_exam'), ('followup', 'followup_lesions', 'followup_exam')):
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(CT / c[exam]['image_ref']))).astype(np.float32)
        mimg = sitk.ReadImage(str(MASKS / f'{pid}_{side}.nii.gz')); m = sitk.GetArrayFromImage(mimg)
        sp = np.array(mimg.GetSpacing()[::-1])                       # z, y, x spacing (mm)
        for l in c[key]:
            lab = int(l['lesion_id'][1:])
            vox = m == lab
            r = dict(patient_id=pid, cohort=c['cohort'], side=side, lesion_id=l['lesion_id'], diameter_mm=l['diameter_mm'],
                     volume_mm3=l.get('volume_mm3'), residual_mm=c['registration']['quality']['residual_mm'])
            for k, nm in (('centroid_mm', 'raw'), ('centroid_registered_mm', 'reg')):
                v = l.get(k) or [np.nan] * 3
                r |= {f'{nm}_x': v[0], f'{nm}_y': v[1], f'{nm}_z': v[2]}
            if vox.sum() >= 3:
                hu = ct[vox]
                r |= dict(hu_mean=float(hu.mean()), hu_sd=float(hu.std()), hu_p10=float(np.percentile(hu, 10)), hu_p90=float(np.percentile(hu, 90)))
                pts = np.argwhere(vox) * sp
                ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T) + 1e-6 * np.eye(3)))
                r['elongation'] = float(np.sqrt(ev[-1] / ev[0]))
            rows.append(r)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = Parallel(n_jobs=48)(delayed(one)(p) for p in sorted(CASES.glob('*.json')))
    df = pd.DataFrame([r for rr in rows for r in rr])
    df.to_csv(OUT / 'lesions.csv', index=False)
    print(df.shape, df.groupby('side').size().to_dict(), 'missing appearance', int(df.hu_mean.isna().sum()))


if __name__ == '__main__':
    main()
