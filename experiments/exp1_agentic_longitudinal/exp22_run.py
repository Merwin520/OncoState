"""Experiment 2.2 downstream pipeline (prediction-only; never opens GT):
rigid registration -> hybrid Hungarian matching -> persistent/disappearing/new -> volume+diameter.
Per patient it writes a structured patient_state.json (the artifact the Code Analysis node reads).

match_fast implements exactly the cost function of hybrid_correspondence.match
(0.45*min(d/30,1)+0.30*(1-IoU)+0.15*relative size difference, 30 mm gate, unmatched cost 0.4,
accept if cost<0.8; organ term unused because no organ labels are supplied), but computes IoU only
for pairs inside the gate and only inside their bounding boxes, which is what makes 100+ candidates
per scan tractable.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('EXP22_OUT', ROOT / 'outputs/exp22_20260918'))
PATHS = {'S0': lambda pid, tp: OUT / f'gtcand/{pid}_{tp}.nii.gz',
         'A': lambda pid, tp: OUT / f'cut_A/{pid}_{tp}.nii.gz',
         'B': lambda pid, tp: OUT / f'cut_B/{pid}_{tp}.nii.gz',
         'P5': lambda pid, tp: OUT / f'cut_P5/{pid}_{tp}.nii.gz',
         'P10': lambda pid, tp: OUT / f'cut_P10/{pid}_{tp}.nii.gz'}


def register(bl_path, fu_path, tfm_path):
    if Path(tfm_path).exists():
        return sitk.ReadTransform(str(tfm_path))
    fixed = sitk.ReadImage(bl_path, sitk.sitkFloat32); moving = sitk.ReadImage(fu_path, sitk.sitkFloat32)
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32); reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.02, 1729); reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift(); reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn(); reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(fixed, moving)
    if not np.isfinite(reg.GetMetricValue()):
        raise RuntimeError('non-finite registration metric')
    Path(tfm_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(tfm_path))
    return tx


def axial_diameter_mm(sub, spacing_zyx):
    """Long-axis diameter on the axial slice with the largest area (convex-hull max pairwise distance)."""
    areas = sub.reshape(sub.shape[0], -1).sum(1)
    k = int(np.argmax(areas))
    pts = np.argwhere(sub[k])
    if len(pts) == 0:
        return 0.0
    xy = pts * np.array([spacing_zyx[1], spacing_zyx[2]])
    try:
        v = xy[ConvexHull(xy).vertices] if len(xy) >= 3 else xy
    except Exception:
        v = xy
    d = np.linalg.norm(v[:, None] - v[None], axis=2).max()
    return float(d + min(spacing_zyx[1], spacing_zyx[2]))


def describe(img):
    arr = sitk.GetArrayFromImage(img)
    sp = img.GetSpacing()[::-1]
    ids = [int(i) for i in np.unique(arr) if i]
    objs = ndimage.find_objects(arr.astype(np.int32))
    vox = float(np.prod(img.GetSpacing()))
    out = {}
    for i in ids:
        sl = objs[i - 1]
        sub = arr[sl] == i
        com = ndimage.center_of_mass(sub)
        idx = [sl[a].start + com[a] for a in range(3)]
        out[i] = {'volume_mm3': float(sub.sum() * vox), 'diameter_mm': axial_diameter_mm(sub, sp),
                  'centroid_mm': list(img.TransformContinuousIndexToPhysicalPoint([float(idx[2]), float(idx[1]), float(idx[0])])),
                  'bbox': [(s.start, s.stop) for s in sl]}
    return arr, out


def match_fast(bl_img, fu_img, tx):
    b_arr, bd = describe(bl_img); f_arr, fd = describe(fu_img)
    warped = sitk.GetArrayFromImage(sitk.Resample(bl_img, fu_img, tx.GetInverse(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt32)) if bd and fd else None
    bids, fids = sorted(bd), sorted(fd)
    n, m = len(bids), len(fids)
    cost = np.full((n + m, n + m), 100.)
    evidence = {}
    if warped is not None:
        wobjs = ndimage.find_objects(warped.astype(np.int32))
        for i, bi in enumerate(bids):
            bc = np.array(tx.TransformPoint(tuple(bd[bi]['centroid_mm'])))
            for j, fj in enumerate(fids):
                d = float(np.linalg.norm(bc - np.array(fd[fj]['centroid_mm'])))
                if d > 30.:
                    continue
                sw = wobjs[bi - 1] if bi - 1 < len(wobjs) else None
                fb = fd[fj]['bbox']
                if sw is None:
                    iou = 0.
                else:
                    lo = [min(sw[a].start, fb[a][0]) for a in range(3)]; hi = [max(sw[a].stop, fb[a][1]) for a in range(3)]
                    sl = tuple(slice(lo[a], hi[a]) for a in range(3))
                    bm = warped[sl] == bi; fm = f_arr[sl] == fj
                    u = np.count_nonzero(bm | fm)
                    iou = float(np.count_nonzero(bm & fm) / u) if u else 0.
                bv, fv = bd[bi]['volume_mm3'], fd[fj]['volume_mm3']
                sd = abs(bv - fv) / max(bv, fv)
                score = .45 * min(d / 30., 1.) + .30 * (1. - iou) + .15 * sd
                cost[i, j] = score
                evidence[i, j] = {'distance_mm': d, 'registered_iou': iou, 'relative_size_difference': sd, 'cost': score}
    cost[:n, m:] = .4; cost[n:, :m] = .4; cost[n:, m:] = 0
    rr, cc = linear_sum_assignment(cost)
    rows = []
    for i, j in zip(rr, cc):
        if i < n and j < m and cost[i, j] < .8:
            rows.append({'baseline_lesion_id': bids[i], 'followup_lesion_id': fids[j], 'status': 'persistent', **evidence[i, j]})
    mb = {r['baseline_lesion_id'] for r in rows}; mf = {r['followup_lesion_id'] for r in rows}
    rows += [{'baseline_lesion_id': i, 'followup_lesion_id': None, 'status': 'disappearing'} for i in bids if i not in mb]
    rows += [{'baseline_lesion_id': None, 'followup_lesion_id': j, 'status': 'newly_appearing'} for j in fids if j not in mf]
    strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk != 'bbox'} for k, v in d.items()}
    return rows, strip(bd), strip(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', choices=list(PATHS), required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    pairs = json.loads((OUT / os.environ.get('EXP22_PAIRS', 'pairs.json')).read_text())[a.shard::a.nshards]
    for p in pairs:
        pid = p['patient_id']
        state_path = OUT / f'state_{a.setting}/{pid}.json'
        if state_path.exists():
            continue
        t0 = time.monotonic()
        rec = {'patient_id': pid, 'setting': a.setting}
        try:
            bp, fp = PATHS[a.setting](pid, 'baseline'), PATHS[a.setting](pid, 'followup')
            if not bp.exists() or not fp.exists():
                raise FileNotFoundError('missing lesion masks')
            tx = register(p['baseline_ct'], p['followup_ct'], OUT / f'reg/{pid}.tfm')
            bl, fu = sitk.ReadImage(str(bp)), sitk.ReadImage(str(fp))
            rows, bd, fd = match_fast(bl, fu, tx)
            rec.update(status='completed', baseline_lesions=bd, followup_lesions=fd, correspondences=rows,
                       units={'volume': 'mm3', 'diameter': 'mm', 'centroid': 'mm LPS'},
                       GT_used=False, runtime_seconds=time.monotonic() - t0)
        except Exception as e:
            rec.update(status='failed', error=str(e))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(rec, indent=1))
        print(a.setting, pid, rec['status'], round(time.monotonic() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
