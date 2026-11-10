"""Batched ULS+ Cut: model loaded once, loops over candidates. Only each candidate's centroid is
used as the prompt (official uclp VOI crop 128x128x64 around it); candidate shapes are never fed
to the network. GPU evidence recorded. Prediction-only; no GT access."""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

ROOT = Path(__file__).resolve().parent
CROP_XYZ = (128, 128, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True); ap.add_argument('--setting', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    sys.path.insert(0, str(ROOT / 'vendor/ULSPlus/preprocessing/uclp'))
    from cropper import compute_crop_region, extract_crop
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    model_dir = ROOT / 'models/ulsplus/Dataset090_ULS23_Combined/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres'
    pred = nnUNetPredictor(perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(model_dir), use_folds=('all',), checkpoint_name='checkpoint_best.pth')
    jobs = [(k, v) for k, v in sorted(json.loads(Path(a.jobs).read_text()).items()) if v['setting'] == a.setting]
    jobs = jobs[a.shard::a.nshards]
    if os.environ.get('EXP22_REVERSE'): jobs = jobs[::-1]      # second wave: work from the end so two waves meet in the middle
    torch.cuda.reset_peak_memory_stats()
    for key, job in jobs:
        out = Path(job['out'])
        if out.exists():
            continue
        t0 = time.monotonic()
        ct = sitk.ReadImage(job['ct']); cand = sitk.ReadImage(job['cand'])
        carr = sitk.GetArrayFromImage(cand)
        result = np.zeros(carr.shape, np.uint16); stats = {}
        for cid in [int(i) for i in np.unique(carr) if i]:
            try:
                zyx = np.argwhere(carr == cid).mean(0)
                phys = ct.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in zyx[::-1]))
                region = compute_crop_region(lesion_centroid=phys, crop_size=CROP_XYZ, image_size=ct.GetSize(),
                                             spacing=ct.GetSpacing(), image=ct)
                crop = extract_crop(ct, region, is_mask=False)
                seg = pred.predict_single_npy_array(sitk.GetArrayFromImage(crop)[None].astype(np.float32),
                                                    {'spacing': list(crop.GetSpacing())[::-1]}, None, None, False)
                seg = np.asarray(seg) > 0
                (sx, sy, sz), (cx, cy, cz) = region[0], CROP_XYZ
                z0, y0, x0 = max(sz, 0), max(sy, 0), max(sx, 0)
                z1, y1, x1 = min(sz + cz, carr.shape[0]), min(sy + cy, carr.shape[1]), min(sx + cx, carr.shape[2])
                sub = seg[z0 - sz:z1 - sz, y0 - sy:y1 - sy, x0 - sx:x1 - sx]
                view = result[z0:z1, y0:y1, x0:x1]
                view[(sub) & (view == 0)] = cid
                stats[cid] = {'voxels': int(sub.sum()), 'status': 'ok'}
            except Exception as e:
                stats[cid] = {'status': 'failed', 'error': str(e)[:200]}
        img = sitk.GetImageFromArray(result); img.CopyInformation(ct)
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out))
        Path(str(out) + '.json').write_text(json.dumps({'key': key, 'gpu': torch.cuda.get_device_name(),
            'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES'), 'runtime_seconds': time.monotonic() - t0,
            'peak_gpu_memory': torch.cuda.max_memory_allocated(), 'candidates': stats, 'GT_used': False}))
        print(key, len(stats), 'candidates', round(time.monotonic() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
