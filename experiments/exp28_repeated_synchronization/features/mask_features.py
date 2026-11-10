"""Local FTV recomputation from the inverse bit-coded UCSF analysis mask."""
import numpy as np

from streaming.seg_header import parse_header


def extract(seg_dataset):
    mask = seg_dataset.pixel_array
    header = parse_header(seg_dataset)
    if mask.ndim != 3 or not header['resampled_voxel_mm']:
        raise ValueError('SEG mask or resampled voxel geometry missing')
    voxel_cc = float(np.prod(header['resampled_voxel_mm']) / 1000.0)
    ftv_voxels = int((mask == 0).sum())
    return mask, header, dict(local_ftv_recomputed_cc=ftv_voxels * voxel_cc,
                              local_ftv_voxels=ftv_voxels, voxel_volume_cc=voxel_cc,
                              mask_shape='x'.join(map(str, mask.shape)))
