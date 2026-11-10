"""Experiment 2.2 baselines, deterministic methods.

Full OncoAgent          : frozen 2.2 setting A state (state_A) -> unified prediction
Reference-mask oracle   : frozen 2.2 setting S0 state (state_S0)
Conventional (centroid) : the repository's registered-centroid Hungarian matcher (MatchingNode / 'registered_centroid_hungarian'):
                          30 mm gate, unmatched cost = gate/2, one-to-one Hungarian. Thresholds are the repository's pre-existing
                          defaults; nothing is tuned on the evaluation cohort. No organ labels exist in this dataset, so no organ constraint.
Conventional (IoU)      : the repository's registered-IoU Hungarian matcher ('registered_iou_hungarian').
All deployable methods use the SAME lesion masks (cut_A), lesion measurements (state_A tables) and frozen registration transform.
"""
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment

import exp22b_common as C
import exp22_run as R

OUT = C.OUT
PAIRS = C.pairs_all()
METHODS = ['Full OncoAgent', 'Reference-mask oracle', 'Conventional (centroid)', 'Conventional (IoU)']


def load_state(setting, pid):
    st = json.loads((OUT / f'state_{setting}/{pid}.json').read_text())
    return st, {int(k): v for k, v in st['baseline_lesions'].items()}, {int(k): v for k, v in st['followup_lesions'].items()}


def hungarian_pairs(dist, gate):
    """Repository MatchingNode assignment: dummy rows/columns with cost gate/2 stop far lesions from being forced into matches."""
    n, m = dist.shape
    costs = np.full((n + m, n + m), gate * 100.)
    costs[:n, :m] = np.where(dist <= gate, dist, gate * 100.)
    costs[:n, m:] = gate / 2; costs[n:, :m] = gate / 2; costs[n:, m:] = 0
    rr, cc = linear_sum_assignment(costs)
    return [(i, j) for i, j in zip(rr, cc) if i < n and j < m and dist[i, j] <= gate]


def centroid_rows(bl, fu, tx, gate=30.):
    bids, fids = sorted(bl), sorted(fu); rows = []
    if bids and fids:
        bc = np.array([tx.TransformPoint(tuple(bl[i]['centroid_mm'])) for i in bids]); fc = np.array([fu[j]['centroid_mm'] for j in fids])
        dist = np.linalg.norm(bc[:, None, :] - fc[None, :, :], axis=2)
        for i, j in hungarian_pairs(dist, gate):
            rows.append({'baseline_lesion_id': bids[i], 'followup_lesion_id': fids[j], 'status': 'persistent', 'distance_mm': float(dist[i, j])})
    return rows


def iou_rows(pid, bl, fu, tx):
    bl_img = sitk.ReadImage(str(R.PATHS['A'](pid, 'baseline'))); fu_img = sitk.ReadImage(str(R.PATHS['A'](pid, 'followup')))
    bids, fids = sorted(bl), sorted(fu); rows = []
    if not (bids and fids): return rows
    warped = sitk.GetArrayFromImage(sitk.Resample(bl_img, fu_img, tx.GetInverse(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt32)); f = sitk.GetArrayFromImage(fu_img)
    sel = (warped > 0) & (f > 0)
    wi, fi = warped[sel].astype(np.int64), f[sel].astype(np.int64)
    P, Q = int(max(bl) + 1), int(max(fu) + 1)
    inter = np.bincount(wi * Q + fi, minlength=P * Q).reshape(P, Q) if wi.size else np.zeros((P, Q))
    sb = np.bincount(warped.ravel().astype(np.int64), minlength=P); sf = np.bincount(f.ravel().astype(np.int64), minlength=Q)
    iou = np.zeros((len(bids), len(fids)))
    for a, i in enumerate(bids):
        for b, j in enumerate(fids):
            if inter[i, j]: iou[a, b] = inter[i, j] / (sb[i] + sf[j] - inter[i, j])
    dist = np.where(iou > 0, 1 - iou, 100.)
    for i, j in hungarian_pairs(dist, 1.):
        rows.append({'baseline_lesion_id': bids[i], 'followup_lesion_id': fids[j], 'status': 'persistent', 'registered_iou': float(1 - dist[i, j])})
    return rows


def one(args):
    pid, method = args
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    p = next(x for x in PAIRS if x['patient_id'] == pid)
    try:
        if method == 'Reference-mask oracle':
            st, bl, fu = load_state('S0', pid); rows = st['correspondences']; masks = 'S0'
        else:
            st, bl, fu = load_state('A', pid); masks = 'A'
            if method == 'Full OncoAgent': rows = st['correspondences']
            else:
                tx = R.register(p['baseline_ct'], p['followup_ct'], OUT / f'reg/{pid}.tfm')
                rows = centroid_rows(bl, fu, tx) if method == 'Conventional (centroid)' else iou_rows(pid, bl, fu, tx)
        ref = C.reference(pid, masks); pred = C.pred_from_rows(rows, bl, fu); ev = C.evaluate(ref, pred)
        return dict(pid=pid, method=method, ev=ev, rows=rows, bl=bl, fu=fu, ref=ref, pred=pred, ok=True)
    except Exception as e:
        return dict(pid=pid, method=method, ev=None, ok=False, err=f'{type(e).__name__}: {e}')


def run(methods=METHODS):
    jobs = [(p['patient_id'], m) for p in PAIRS for m in methods]
    with Pool(16) as pool:
        return pool.map(one, jobs, chunksize=2)


if __name__ == '__main__':
    res = run()
    print(len(res), 'evaluations;', sum(not r['ok'] for r in res), 'failed')
