"""Geometry validation, compact PE features and de-identified figure thumbnail."""
from pathlib import Path

import numpy as np
from PIL import Image


def stack_series(datasets):
    ordered = sorted(datasets, key=lambda value: (float(value.get('InstanceNumber', 0)),
                                                   str(value.get('SOPInstanceUID', ''))))
    return np.stack([value.pixel_array for value in ordered])


def spatial_slice_count(datasets):
    positions = {str(value.get('ImagePositionPatient', '')) for value in datasets}
    return len(positions) if positions != {''} else len(datasets)


def extract(mask, pe_datasets, dce_datasets):
    pe = stack_series(pe_datasets)
    pe_match = pe.shape == mask.shape
    dce_match = bool(dce_datasets) and dce_datasets[0].Rows == mask.shape[1] and dce_datasets[0].Columns == mask.shape[2]
    dce_match = dce_match and spatial_slice_count(dce_datasets) == mask.shape[0]
    tumour = mask == 0
    values = pe[tumour] if pe_match else np.asarray([])
    return dict(pe_geometry_match=bool(pe_match), dce_geometry_match=bool(dce_match),
                pe_tumour_mean=None if not len(values) else float(values.mean()),
                pe_tumour_sd=None if not len(values) else float(values.std()),
                pe_shape='x'.join(map(str, pe.shape)), dce_instances=len(dce_datasets))


def save_thumbnail(mask, dce_datasets, path):
    z = mask.shape[0] // 2
    slices = [value for value in dce_datasets if int(float(value.get('InstanceNumber', 1))) <= mask.shape[0]]
    if len(slices) < mask.shape[0]:
        slices = sorted(dce_datasets, key=lambda value: float(value.get('InstanceNumber', 0)))[:mask.shape[0]]
    image = stack_series(slices)[z].astype(float)
    low, high = np.percentile(image, [1, 99])
    gray = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    overlay = mask[z] == 0
    rgb[overlay, 0] = 255
    rgb[overlay, 1:] = (rgb[overlay, 1:] * .35).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
