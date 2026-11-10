"""Experiment 2.2 (radiologist-position protocol), extended evaluation. GT is opened only here, after predictions are persisted.

Adds to exp22_eval.py: per-lesion Dice/IoU/HD95/volume ratio, per-lesion change class (enlarged/stable/shrunk at +-20% volume),
patient-level tumour-burden and count errors, and a registration/correspondence sanity check on GT shared-id pairs.
Lesion mapping and identity reference are exactly those of exp22_eval.py (one-to-one Hungarian on IoU > 0.1; shared GT id across
timepoints = same lesion, an empirical reference the dataset README does not define).
"""
import csv, json, os, sys
from multiprocessing import Pool
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import exp22_run as R
import exp22_eval as E

OUT = R.OUT
CLASSES = E.CLASSES
SETTINGS = os.environ.get('EXP22_SETTINGS', 'S0,A').split(',')
CHG = 20.0  # % volume change threshold for enlarged / shrunk


def hd95_mm(a, b, spacing_zyx):
    """95th-percentile symmetric surface distance between two boolean crops (same grid)."""
    if not a.any() or not b.any():
        return np.nan
    sa = a & ~ndimage.binary_erosion(a); sb = b & ~ndimage.binary_erosion(b)
    if not sa.any(): sa = a
    if not sb.any(): sb = b
    da = ndimage.distance_transform_edt(~sa, sampling=spacing_zyx)
    db = ndimage.distance_transform_edt(~sb, sampling=spacing_zyx)
    d = np.concatenate([db[sa], da[sb]])
    return float(np.percentile(d, 95))


def obj_metrics(parr, pid_, garr, gid, spacing_zyx, vox_mm3):
    """Overlap metrics of predicted object pid_ vs GT object gid inside their joint bounding box (+ margin)."""
    pm, gm = parr == pid_, garr == gid
    idx = np.argwhere(pm | gm)
    lo = np.maximum(idx.min(0) - 3, 0); hi = idx.max(0) + 4
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    p, g = pm[sl], gm[sl]
    inter = int((p & g).sum()); un = int((p | g).sum())
    return dict(dice=2 * inter / (p.sum() + g.sum()), iou=inter / un if un else np.nan,
                hd95_mm=hd95_mm(p, g, spacing_zyx), vol_ratio=float(p.sum() / g.sum()))


def change_class(pct):
    return 'enlarged' if pct > CHG else 'shrunk' if pct < -CHG else 'stable'


def one_patient(args):
    setting, p = args
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    pid = p['patient_id']
    sp = OUT / f'state_{setting}/{pid}.json'
    if not sp.exists():
        return [], dict(setting=setting, patient_id=pid, cohort=p.get('cohort'), status='not_run'), []
    st = json.loads(sp.read_text())
    if st.get('status') != 'completed':
        return [], dict(setting=setting, patient_id=pid, cohort=p.get('cohort'), status='failed', error=str(st.get('error'))[:120]), []
    gb, gf = E.gt_img(pid, 'baseline'), E.gt_img(pid, 'followup')
    garr_b, garr_f = sitk.GetArrayFromImage(gb), sitk.GetArrayFromImage(gf)
    gd_b, gd_f = R.describe(gb)[1], R.describe(gf)[1]
    sp_b, sp_f = gb.GetSpacing()[::-1], gf.GetSpacing()[::-1]
    pa_b = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting](pid, 'baseline'))))
    pa_f = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting](pid, 'followup'))))
    mb, _, _, _ = E.overlap_map(pa_b, garr_b); mf, _, _, _ = E.overlap_map(pa_f, garr_f)
    inv_b = {g: q for q, g in mb.items()}; inv_f = {g: q for q, g in mf.items()}
    gbl, gfu = set(gd_b), set(gd_f)
    gclass = {g: 'persistent' if g in gbl and g in gfu else 'disappearing' if g in gbl else 'newly_appearing' for g in gbl | gfu}
    rows_c = st['correspondences']
    row_of_b = {r['baseline_lesion_id']: r for r in rows_c if r['baseline_lesion_id'] is not None}
    row_of_f = {r['followup_lesion_id']: r for r in rows_c if r['followup_lesion_id'] is not None}
    bl_les = {int(k): v for k, v in st['baseline_lesions'].items()}; fu_les = {int(k): v for k, v in st['followup_lesions'].items()}
    correct_pair_of, n_pred_pairs, tp_pairs = {}, 0, 0
    for r in rows_c:
        if r['status'] != 'persistent':
            continue
        n_pred_pairs += 1
        a, b = r['baseline_lesion_id'], r['followup_lesion_id']
        if a in mb and b in mf and mb[a] == mf[b]:
            tp_pairs += 1; correct_pair_of[mb[a]] = r
    false_new = sum(1 for r in rows_c if r['status'] == 'newly_appearing' and (r['followup_lesion_id'] not in mf or mf[r['followup_lesion_id']] in gbl))
    false_dis = sum(1 for r in rows_c if r['status'] == 'disappearing' and (r['baseline_lesion_id'] not in mb or mb[r['baseline_lesion_id']] in gfu))
    vox_b, vox_f = float(np.prod(gb.GetSpacing())), float(np.prod(gf.GetSpacing()))
    lesion_rows = []
    for g, cls in sorted(gclass.items()):
        pb, pf = inv_b.get(g), inv_f.get(g)
        if cls == 'persistent':
            r_ = row_of_b.get(pb) if pb is not None else (row_of_f.get(pf) if pf is not None else None)
        elif cls == 'disappearing':
            r_ = row_of_b.get(pb) if pb is not None else None
        else:
            r_ = row_of_f.get(pf) if pf is not None else None
        row = dict(setting=setting, patient_id=pid, cohort=p.get('cohort'), gt_id=g, gt_class=cls, detected=pb is not None or pf is not None,
                   pred_status=r_['status'] if r_ else 'undetected')
        row.update(gt_vol_bl_mL=gd_b[g]['volume_mm3'] / 1000 if g in gd_b else None, gt_vol_fu_mL=gd_f[g]['volume_mm3'] / 1000 if g in gd_f else None,
                   gt_diam_bl_mm=gd_b[g]['diameter_mm'] if g in gd_b else None, gt_diam_fu_mm=gd_f[g]['diameter_mm'] if g in gd_f else None)
        for tag, q, les, garr, parr, spc, vx, gd in (('bl', pb, bl_les, garr_b, pa_b, sp_b, vox_b, gd_b), ('fu', pf, fu_les, garr_f, pa_f, sp_f, vox_f, gd_f)):
            if q is not None and g in gd:
                row[f'pred_vol_{tag}_mL'] = les[q]['volume_mm3'] / 1000; row[f'pred_diam_{tag}_mm'] = les[q]['diameter_mm']
                for k, v in obj_metrics(parr, q, garr, g, spc, vx).items():
                    row[f'{k}_{tag}'] = v
        if cls == 'persistent' and pb is not None and pf is not None:
            ok = g in correct_pair_of
            row['pair_correct'] = ok
            gvb, gvf = gd_b[g]['volume_mm3'], gd_f[g]['volume_mm3']
            row['gt_change_pct'] = (gvf - gvb) / gvb * 100
            row['gt_change_class'] = change_class(row['gt_change_pct'])
            if ok:
                pvb, pvf = bl_les[pb]['volume_mm3'], fu_les[pf]['volume_mm3']
                row['pred_change_pct'] = (pvf - pvb) / pvb * 100
                row['pred_change_class'] = change_class(row['pred_change_pct'])
        lesion_rows.append(row)
    # patient-level burden / count errors (all predicted lesions vs all GT lesions)
    tot = lambda d: sum(v['volume_mm3'] for v in d.values()) / 1000
    pred_cnt = {c: sum(1 for r in rows_c if r['status'] == c) for c in CLASSES}
    gt_cnt = {c: sum(1 for v in gclass.values() if v == c) for c in CLASSES}
    prow = dict(setting=setting, patient_id=pid, cohort=p.get('cohort'), status='completed', gt_bl=len(gbl), gt_fu=len(gfu),
                gt_persistent=gt_cnt['persistent'], gt_new=gt_cnt['newly_appearing'], gt_disappeared=gt_cnt['disappearing'],
                pred_bl=len(bl_les), pred_fu=len(fu_les), pred_pairs=n_pred_pairs, correct_pairs=tp_pairs, false_new_lesions=false_new,
                false_disappeared_lesions=false_dis, pred_new=pred_cnt['newly_appearing'], pred_disappeared=pred_cnt['disappearing'],
                gt_burden_bl_mL=tot(gd_b), gt_burden_fu_mL=tot(gd_f), pred_burden_bl_mL=tot(bl_les), pred_burden_fu_mL=tot(fu_les),
                runtime_seconds=st.get('runtime_seconds'))
    # registration / correspondence sanity (only once, S0: GT shapes; independent of Cut)
    reg_rows = []
    if setting == 'S0':
        tx = R.register(p['baseline_ct'], p['followup_ct'], OUT / f'reg/{pid}.tfm')
        shared = sorted(gbl & gfu)
        for g in shared:
            bc = np.array(tx.TransformPoint(tuple(gd_b[g]['centroid_mm'])))
            d_same = float(np.linalg.norm(bc - np.array(gd_f[g]['centroid_mm'])))
            others = [np.linalg.norm(bc - np.array(gd_f[h]['centroid_mm'])) for h in gfu if h != g]
            reg_rows.append(dict(patient_id=pid, cohort=p.get('cohort'), gt_id=g, dist_shared_id_mm=d_same,
                                 dist_nearest_other_fu_lesion_mm=float(min(others)) if others else None))
    return lesion_rows, prow, reg_rows


def main():
    pairs = json.loads((OUT / os.environ.get('EXP22_PAIRS', 'pairs.json')).read_text())
    jobs = [(s, p) for s in SETTINGS for p in pairs]
    with Pool(16) as pool:
        res = pool.map(one_patient, jobs, chunksize=2)
    for name, idx in (('lesions', 0), ('patients', 1), ('regsanity', 2)):
        rows = [r for x in res for r in (x[idx] if idx != 1 else [x[idx]])]
        if name == 'regsanity':
            if not rows: continue
            path = OUT / 'eval_regsanity_S0.csv'
            keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(open(path, 'w', newline=''), fieldnames=keys); w.writeheader(); w.writerows(rows)
            continue
        for s in SETTINGS:
            rs = [r for r in rows if r['setting'] == s]
            keys = sorted({k for r in rs for k in r})
            with open(OUT / f'evalx_{name}_{s}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rs)
    print('done', len(res))


if __name__ == '__main__':
    main()
