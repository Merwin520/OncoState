"""Experiment 2.2 evaluation (GT opened only here, after all predictions are persisted).

Lesion mapping = same rule as evaluation.compare(): connected label objects, one-to-one Hungarian,
IoU > 0.1. GT identity reference = lesion id shared across timepoints in the Longitudinal-CT masks
(empirical reference; the dataset README does not define id semantics). Merging is not evaluated.
"""
import os
import argparse, csv, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

import exp22_run as R

OUT = R.OUT
DATA = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'Longitudinal-CT'))
CLASSES = ['persistent', 'disappearing', 'newly_appearing']


def gt_img(pid, tp):
    pre, folder = ('BL', 'inputsTr') if tp == 'baseline' else ('FU', 'targetsTr')
    p = DATA / folder / f'{pid}_{pre}_mask_{pre}_img_00.nii.gz'
    return sitk.ReadImage(str(p)) if p.is_file() else None


def overlap_map(pred_arr, gt_arr, thr=0.1):
    """Returns (mapping pred_id->gt_id, dice per mapped pair, sizes)."""
    sel = (pred_arr > 0) | (gt_arr > 0)
    pl = pred_arr[sel].astype(np.int64); gl = gt_arr[sel].astype(np.int64)
    P, G = int(pl.max()) + 1 if pl.size else 1, int(gl.max()) + 1 if gl.size else 1
    ov = np.bincount(pl * G + gl, minlength=P * G).reshape(P, G)
    ps = np.bincount(pl, minlength=P); gs = np.bincount(gl, minlength=G)
    pids = [i for i in range(1, P) if ps[i]]; gids = [i for i in range(1, G) if gs[i]]
    if not pids or not gids:
        return {}, {}, ps, gs
    iou = np.zeros((len(pids), len(gids)))
    for a, i in enumerate(pids):
        for b, j in enumerate(gids):
            if ov[i, j]:
                iou[a, b] = ov[i, j] / (ps[i] + gs[j] - ov[i, j])
    rr, cc = linear_sum_assignment(-(iou > thr).astype(float) - iou * 1e-3)
    mp, dice = {}, {}
    for a, b in zip(rr, cc):
        if iou[a, b] > thr:
            mp[pids[a]] = gids[b]
            dice[pids[a]] = float(2 * ov[pids[a], gids[b]] / (ps[pids[a]] + gs[gids[b]]))
    return mp, dice, ps, gs


def safe_div(a, b):
    return a / b if b else None


def evaluate_setting(setting, pairs):
    lesion_rows, patient_rows = [], []
    pooled = dict(pair_tp=0, pair_fp=0, pair_fn=0, both_det=0, both_det_correct=0, find_tp=0, find_fp=0, find_fn=0)
    y_true, y_pred = [], []
    vol_err, vol_rel, diam_err, chg_err, chg_sign, dices = [], [], [], [], [], []
    completed = 0
    for p in pairs:
        pid = p['patient_id']
        sp = OUT / f'state_{setting}/{pid}.json'
        if not sp.exists():
            patient_rows.append({'setting': setting, 'patient_id': pid, 'status': 'not_run'}); continue
        st = json.loads(sp.read_text())
        if st.get('status') != 'completed':
            patient_rows.append({'setting': setting, 'patient_id': pid, 'status': 'failed', 'error': st.get('error')}); continue
        completed += 1
        gb, gf = gt_img(pid, 'baseline'), gt_img(pid, 'followup')
        garr_b, garr_f = sitk.GetArrayFromImage(gb), sitk.GetArrayFromImage(gf)
        gd_b, gd_f = R.describe(gb)[1], R.describe(gf)[1]
        pa_b = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting](pid, 'baseline'))))
        pa_f = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting](pid, 'followup'))))
        mb, db, _, _ = overlap_map(pa_b, garr_b); mf, df, _, _ = overlap_map(pa_f, garr_f)
        inv_b = {g: p_ for p_, g in mb.items()}; inv_f = {g: p_ for p_, g in mf.items()}
        gbl, gfu = set(gd_b), set(gd_f)
        gclass = {}
        for g in gbl | gfu:
            gclass[g] = 'persistent' if g in gbl and g in gfu else 'disappearing' if g in gbl else 'newly_appearing'
        rows = st['correspondences']
        row_of_b = {r['baseline_lesion_id']: r for r in rows if r['baseline_lesion_id'] is not None}
        row_of_f = {r['followup_lesion_id']: r for r in rows if r['followup_lesion_id'] is not None}
        bl_les = {int(k): v for k, v in st['baseline_lesions'].items()}; fu_les = {int(k): v for k, v in st['followup_lesions'].items()}
        # pair-level matching
        n_pred_pairs = tp_pairs = 0
        correct_pair_of = {}
        for r in rows:
            if r['status'] != 'persistent':
                continue
            n_pred_pairs += 1
            a, b = r['baseline_lesion_id'], r['followup_lesion_id']
            if a in mb and b in mf and mb[a] == mf[b]:
                tp_pairs += 1; correct_pair_of[mb[a]] = r
        gt_pers = [g for g, c in gclass.items() if c == 'persistent']
        pooled['pair_tp'] += tp_pairs; pooled['pair_fp'] += n_pred_pairs - tp_pairs; pooled['pair_fn'] += len(gt_pers) - tp_pairs
        false_new = sum(1 for r in rows if r['status'] == 'newly_appearing' and (r['followup_lesion_id'] not in mf or mf[r['followup_lesion_id']] in gbl))
        # per GT-lesion status, detection, cut dice, measurement
        for g, cls in sorted(gclass.items()):
            pb, pf = inv_b.get(g), inv_f.get(g)
            detected = pb is not None or pf is not None
            if cls == 'persistent':
                r_ = row_of_b.get(pb) if pb is not None else (row_of_f.get(pf) if pf is not None else None)
            elif cls == 'disappearing':
                r_ = row_of_b.get(pb) if pb is not None else None
            else:
                r_ = row_of_f.get(pf) if pf is not None else None
            pred_status = r_['status'] if r_ else 'undetected'
            y_true.append(cls); y_pred.append(pred_status)
            row = {'setting': setting, 'patient_id': pid, 'gt_id': g, 'gt_class': cls, 'detected': detected,
                   'pred_status': pred_status}
            gv_b = gd_b[g]['volume_mm3'] / 1000 if g in gd_b else None
            gv_f = gd_f[g]['volume_mm3'] / 1000 if g in gd_f else None
            row.update(gt_vol_bl_mL=gv_b, gt_vol_fu_mL=gv_f,
                       gt_diam_bl_mm=gd_b[g]['diameter_mm'] if g in gd_b else None, gt_diam_fu_mm=gd_f[g]['diameter_mm'] if g in gd_f else None)
            for tag, pid_, les, gd, dcs in (('bl', pb, bl_les, gd_b, db), ('fu', pf, fu_les, gd_f, df)):
                if pid_ is not None and g in gd:
                    pv = les[pid_]['volume_mm3'] / 1000; gv = gd[g]['volume_mm3'] / 1000
                    row[f'pred_vol_{tag}_mL'] = pv; row[f'pred_diam_{tag}_mm'] = les[pid_]['diameter_mm']
                    row[f'cut_dice_{tag}'] = dcs[pid_]
                    dices.append(dcs[pid_]); vol_err.append(abs(pv - gv)); vol_rel.append(abs(pv - gv) / gv if gv else np.nan)
                    diam_err.append(abs(les[pid_]['diameter_mm'] - gd[g]['diameter_mm']))
            if cls == 'persistent' and pb is not None and pf is not None:
                pooled['both_det'] += 1
                ok = g in correct_pair_of
                pooled['both_det_correct'] += int(ok)
                row['pair_correct'] = ok
                if ok:
                    pv_b, pv_f = bl_les[pb]['volume_mm3'], fu_les[pf]['volume_mm3']
                    gvb, gvf = gd_b[g]['volume_mm3'], gd_f[g]['volume_mm3']
                    if pv_b and gvb:
                        pc, gc = (pv_f - pv_b) / pv_b * 100, (gvf - gvb) / gvb * 100
                        row['pred_change_pct'] = pc; row['gt_change_pct'] = gc
                        chg_err.append(abs(pc - gc)); chg_sign.append(np.sign(pc) == np.sign(gc))
            lesion_rows.append(row)
        # Find-stage detection (B: before Cut) using the raw candidates
        if setting == 'B':
            for tp_, garr in (('baseline', garr_b), ('followup', garr_f)):
                fc = sitk.GetArrayFromImage(sitk.ReadImage(str(OUT / f'findcand/{pid}_{tp_}.nii.gz')))
                m_, _, ps_, _ = overlap_map(fc, garr)
                n_c = int((ps_[1:] > 0).sum()); n_g = len({int(v) for v in np.unique(garr) if v})
                pooled['find_tp'] += len(m_); pooled['find_fp'] += n_c - len(m_); pooled['find_fn'] += n_g - len(m_)
        patient_rows.append({'setting': setting, 'patient_id': pid, 'status': 'completed', 'gt_bl': len(gbl), 'gt_fu': len(gfu),
                             'gt_persistent': len(gt_pers), 'pred_bl': len(bl_les), 'pred_fu': len(fu_les),
                             'pred_pairs': n_pred_pairs, 'correct_pairs': tp_pairs, 'false_new_lesions': false_new})
    n_gt = len(y_true); det = [i for i, v in enumerate(y_pred) if v != 'undetected']
    cls_recall = {c: safe_div(sum(1 for t, p_ in zip(y_true, y_pred) if t == c and p_ == c), sum(1 for t in y_true if t == c)) for c in CLASSES}
    pt = pooled
    prec = safe_div(pt['pair_tp'], pt['pair_tp'] + pt['pair_fp']); rec = safe_div(pt['pair_tp'], pt['pair_tp'] + pt['pair_fn'])
    summary = {
        'patients_attempted': len(pairs), 'task_completion_rate': completed / len(pairs),
        'gt_lesions': n_gt, 'gt_lesions_detected': len(det), 'lesion_sensitivity_final': safe_div(len(det), n_gt),
        'cut_dice_mean_on_detected': float(np.mean(dices)) if dices else None,
        'match_pair_precision': prec, 'match_pair_recall': rec,
        'match_pair_f1': safe_div(2 * prec * rec, prec + rec) if prec and rec else 0.0,
        'match_accuracy_given_both_detected': safe_div(pt['both_det_correct'], pt['both_det']),
        'status_accuracy_all_gt_lesions': safe_div(sum(t == p_ for t, p_ in zip(y_true, y_pred)), n_gt),
        'status_accuracy_given_detected': safe_div(sum(y_true[i] == y_pred[i] for i in det), len(det)),
        'persistent_recall': cls_recall['persistent'], 'disappearing_sensitivity': cls_recall['disappearing'],
        'new_lesion_sensitivity': cls_recall['newly_appearing'],
        'status_weighted_f1': float(f1_score(y_true, y_pred, labels=CLASSES, average='weighted', zero_division=0)) if n_gt else None,
        'volume_MAE_mL': float(np.mean(vol_err)) if vol_err else None,
        'volume_median_relative_error': float(np.nanmedian(vol_rel)) if vol_rel else None,
        'diameter_MAE_mm': float(np.mean(diam_err)) if diam_err else None,
        'volume_change_MAE_pct_points': float(np.mean(chg_err)) if chg_err else None,
        'volume_change_direction_agreement': float(np.mean(chg_sign)) if chg_sign else None,
        'n_correct_persistent_pairs_measured': len(chg_err),
        'false_new_lesions_per_patient': float(np.mean([r['false_new_lesions'] for r in patient_rows if r['status'] == 'completed'])) if completed else None,
        'predicted_lesions_per_patient_bl': float(np.mean([r['pred_bl'] for r in patient_rows if r['status'] == 'completed'])) if completed else None,
        'gt_lesions_per_patient_bl': float(np.mean([r['gt_bl'] for r in patient_rows if r['status'] == 'completed'])) if completed else None,
    }
    if setting == 'B':
        summary.update(find_sensitivity=safe_div(pt['find_tp'], pt['find_tp'] + pt['find_fn']),
                       find_fp_per_scan=pt['find_fp'] / (2 * completed) if completed else None)
    return summary, lesion_rows, patient_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', nargs='+', default=['S0', 'A', 'B'])
    a = ap.parse_args()
    pairs = json.loads((OUT / 'pairs.json').read_text())
    result = {}
    for s in a.settings:
        summ, lrows, prows = evaluate_setting(s, pairs)
        result[s] = summ
        for name, rows in (('lesions', lrows), ('patients', prows)):
            keys = sorted({k for r in rows for k in r})
            with open(OUT / f'eval_{name}_{s}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(s, json.dumps(summ, indent=1), flush=True)
    (OUT / 'exp22_summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
