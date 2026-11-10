"""Official IDC metadata (idc-index) for I-SPY collections. Metadata only: no image bytes are read here."""
import json

import pandas as pd

from paths import metadata_dir


def load_series(collection='ispy2', refresh=False):
    """One row per series from the IDC index, cached as parquet with the IDC data version recorded."""
    f = metadata_dir() / f'idc_{collection}_series.parquet'
    meta = metadata_dir() / f'idc_{collection}_series.json'
    if refresh or not f.exists():
        from idc_index import IDCClient
        c = IDCClient()
        d = c.index[c.index.collection_id == collection].copy()
        d.to_parquet(f)
        meta.write_text(json.dumps({'idc_version': c.get_idc_version(), 'collection_id': collection, 'n_series': len(d)}))
    return pd.read_parquet(f), json.loads(meta.read_text()) if meta.exists() else {}


def derived_kind(desc):
    """Classify I-SPY2 UCSF derived objects by series description (ordered: 'VOLSER' itself contains 'SER')."""
    s = str(desc)
    lat = 'bilateral' if 'bi-lateral' in s else 'unilateral_cropped' if 'uni-lateral cropped' in s else None
    if 'VOLSER' not in s:
        return 'acquired', None
    if s.endswith('Analysis Mask'):
        return 'analysis_mask_seg', lat
    if s.endswith(': SER'):
        return 'ser_map', lat
    if 'original DCE' in s:
        return 'cropped_dce' if lat == 'unilateral_cropped' else 'derived_dce', lat
    if ': PE' in s:
        return 'pe_map', lat
    return 'other_derived', lat
