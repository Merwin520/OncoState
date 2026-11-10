"""Experiment 2.2 external-baseline comparison: shared helpers.

Everything here reuses the frozen Experiment 2.2 definitions (exp22_run / exp22_eval / exp22_full_eval):
  * lesion correspondence  : predicted pair (a, b) is correct if both lesions map (Hungarian, IoU > 0.1) to the same reference lesion;
                             reference identity = lesion id shared by baseline and follow-up masks;
  * longitudinal status    : persistent (matched pair) / disappearing (unmatched baseline) / newly_appearing (unmatched follow-up);
  * lesion change class    : enlarged / stable / shrunk at +-20% volume change of a matched pair;
  * patient burden         : direction of the change in total lesion volume (increase / decrease / no_change).
Every method is converted to one prediction schema and evaluated by the same function.
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/exp22_baselines'
os.environ.setdefault('EXP22_OUT', str(BASE / 'exp22_regen'))   # must be set before exp22_run is imported
os.environ.setdefault('EXP22_ALL', '1')

import numpy as np
import SimpleITK as sitk

import exp22_run as R
import exp22_eval as E

OUT = R.OUT
CHG = 20.0
STATUS = ['persistent', 'disappearing', 'newly_appearing']
CHANGE = ['enlarged', 'stable', 'shrunk']
BURDEN = ['increase', 'decrease', 'no_change']


def change_class(pct):
    return 'enlarged' if pct > CHG else 'shrunk' if pct < -CHG else 'stable'


def burden_dir(bl, fu):
    return 'increase' if fu > bl else 'decrease' if fu < bl else 'no_change'


def pairs_all():
    return json.loads((OUT / 'pairs.json').read_text())


def reference(pid, setting_masks):
    """Reference structure of one patient plus the pred->reference lesion maps for the masks the method used."""
    gb, gf = E.gt_img(pid, 'baseline'), E.gt_img(pid, 'followup')
    garr_b, garr_f = sitk.GetArrayFromImage(gb), sitk.GetArrayFromImage(gf)
    gd_b, gd_f = R.describe(gb)[1], R.describe(gf)[1]
    pa_b = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting_masks](pid, 'baseline'))))
    pa_f = sitk.GetArrayFromImage(sitk.ReadImage(str(R.PATHS[setting_masks](pid, 'followup'))))
    mb, _, _, _ = E.overlap_map(pa_b, garr_b); mf, _, _, _ = E.overlap_map(pa_f, garr_f)
    gbl, gfu = set(gd_b), set(gd_f)
    gclass = {g: 'persistent' if g in gbl and g in gfu else 'disappearing' if g in gbl else 'newly_appearing' for g in gbl | gfu}
    gchg = {g: change_class((gd_f[g]['volume_mm3'] - gd_b[g]['volume_mm3']) / gd_b[g]['volume_mm3'] * 100) for g in gbl & gfu if gd_b[g]['volume_mm3']}
    burden = burden_dir(sum(v['volume_mm3'] for v in gd_b.values()), sum(v['volume_mm3'] for v in gd_f.values()))
    return dict(mb=mb, mf=mf, inv_b={g: p for p, g in mb.items()}, inv_f={g: p for p, g in mf.items()}, gclass=gclass, gchg=gchg, gburden=burden,
                gt_vol_b={g: v['volume_mm3'] for g, v in gd_b.items()}, gt_vol_f={g: v['volume_mm3'] for g, v in gd_f.items()},
                gt_diam_b={g: v['diameter_mm'] for g, v in gd_b.items()}, gt_diam_f={g: v['diameter_mm'] for g, v in gd_f.items()},
                gt_cent_b={g: v['centroid_mm'] for g, v in gd_b.items()}, gt_cent_f={g: v['centroid_mm'] for g, v in gd_f.items()})


def pred_from_rows(rows, bl_les, fu_les):
    """Unified prediction schema from a correspondence table (rows) and lesion tables (id -> volume/diameter/centroid)."""
    pairs = [(r['baseline_lesion_id'], r['followup_lesion_id']) for r in rows if r['status'] == 'persistent']
    mb = {a for a, _ in pairs}; mf = {b for _, b in pairs}
    status_b = {i: ('persistent' if i in mb else 'disappearing') for i in bl_les}
    status_f = {i: ('persistent' if i in mf else 'newly_appearing') for i in fu_les}
    change = {(a, b): change_class((fu_les[b]['volume_mm3'] - bl_les[a]['volume_mm3']) / bl_les[a]['volume_mm3'] * 100) for a, b in pairs if bl_les[a]['volume_mm3']}
    burden = burden_dir(sum(v['volume_mm3'] for v in bl_les.values()), sum(v['volume_mm3'] for v in fu_les.values()))
    return dict(pairs=pairs, status_b=status_b, status_f=status_f, change=change, burden=burden, success=True)


def evaluate(ref, pred):
    """Per-patient sufficient statistics under the frozen definitions (all counts, so they pool and bootstrap exactly)."""
    inv_b, inv_f, mb, mf = ref['inv_b'], ref['inv_f'], ref['mb'], ref['mf']
    out = dict(tp=0, pred_pairs=0, gt_pers=0, conf=np.zeros((3, 4)), chg_conf=np.zeros((3, 3)), chg_cond_n=0, chg_cond_ok=0, chg_unc_n=0, chg_unc_ok=0, burden_ok=None,
               burden_pred=pred.get('burden'), burden_gt=ref['gburden'], burden_conf=np.zeros((3, 4)))
    pairs = pred.get('pairs') or []
    out['pred_pairs'] = len(pairs)
    correct_of = {}
    for a, b in pairs:
        if a in mb and b in mf and mb[a] == mf[b]:
            out['tp'] += 1; correct_of[mb[a]] = (a, b)
    gt_pers = [g for g, c in ref['gclass'].items() if c == 'persistent']
    out['gt_pers'] = len(gt_pers)
    sb, sf = pred.get('status_b') or {}, pred.get('status_f') or {}
    for g, cls in ref['gclass'].items():
        pb, pf = inv_b.get(g), inv_f.get(g)
        if cls == 'persistent':
            st = sb.get(pb) if pb is not None and pb in sb else (sf.get(pf) if pf is not None and pf in sf else None)
        elif cls == 'disappearing':
            st = sb.get(pb) if pb is not None else None
        else:
            st = sf.get(pf) if pf is not None else None
        col = STATUS.index(st) if st in STATUS else 3
        out['conf'][STATUS.index(cls), col] += 1
        if cls == 'persistent' and pb is not None and pf is not None:
            out['chg_unc_n'] += 1
            if g in correct_of and g in ref['gchg']:
                out['chg_cond_n'] += 1
                pc = pred['change'].get(correct_of[g]); ok = pc == ref['gchg'][g]
                if pc in CHANGE: out['chg_conf'][CHANGE.index(ref['gchg'][g]), CHANGE.index(pc)] += 1
                out['chg_cond_ok'] += int(ok); out['chg_unc_ok'] += int(ok)
    out['burden_ok'] = None if pred.get('burden') is None else int(pred['burden'] == ref['gburden'])
    out['burden_conf'][BURDEN.index(ref['gburden']), BURDEN.index(pred['burden']) if pred.get('burden') in BURDEN else 3] += 1
    return out
