#!/usr/bin/env python3
"""Freeze the downloaded MU-Glioma-Post release: convert the clinical XLSX to CSV (unchanged content) and write a sha256 inventory of
every released file. Needs openpyxl (system python3); the project runtime lacks it.   python3 freeze_sources.py"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get('MU_GLIOMA_ROOT', os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'MU-Glioma-Post')))
OUT = Path(__file__).resolve().parents[2] / 'outputs/multivisit_longitudinal'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = ROOT / 'raw/MU-Glioma-Post_ClinicalData-July2025.xlsx'
    pd.read_excel(x, sheet_name='MU Glioma Post').to_csv(ROOT / 'raw/clinical.csv', index=False)
    inv = {'raw': {p.name: sha(p) for p in sorted((ROOT / 'raw').iterdir()) if p.is_file()}, 'nifti_masks': {}, 'nifti_file_count': 0, 'nifti_bytes': 0}
    for p in sorted((ROOT / 'nifti').rglob('*.nii.gz')):
        inv['nifti_file_count'] += 1
        inv['nifti_bytes'] += p.stat().st_size
        if p.name.endswith('_tumorMask.nii.gz'):
            inv['nifti_masks'][str(p.relative_to(ROOT))] = sha(p)
    (OUT / 'source_inventory.json').write_text(json.dumps(inv, indent=1))
    print('files', inv['nifti_file_count'], 'GB', round(inv['nifti_bytes'] / 1e9, 2), 'masks', len(inv['nifti_masks']))


if __name__ == '__main__':
    main()
