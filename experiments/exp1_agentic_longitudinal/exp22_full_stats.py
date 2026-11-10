"""Experiment 2.2 (radiologist-position protocol): extended metrics with patient-level bootstrap.

Main cohort = 'locked' patients (never used for tool selection); the 25 'dev' patients are reported separately (point estimates only).
Resampling unit = patient (2000 draws, same draws for S0 and A so S0->A differences are paired).
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = Path(os.environ.get('EXP22_OUT', ROOT / 'outputs/exp22_baselines/exp22_regen'))   # regenerated S0/A outputs (regen_chain)
DST = ROOT / 'outputs/exp22_full_final'
DST.mkdir(parents=True, exist_ok=True)
SETTINGS = ['S0', 'A']
CLASSES = ['persistent', 'disappearing', 'newly_appearing']
CHG = ['enlarged', 'stable', 'shrunk']
NB, SEED = 2000, 20260919


def kappa(conf):
    n = conf.sum()
    if n == 0:
        return np.nan
    po = np.trace(conf) / n; pe = (conf.sum(0) * conf.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def f1_from_conf(conf):
    out = []
    for c in range(3):
        tp = conf[c, c]; fp = conf[:3, c].sum() - tp; fn = conf[c].sum() - tp
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
    return np.array(out)


def load(setting):
    L = pd.read_csv(SRC / f'evalx_lesions_{setting}.csv')
    P = pd.read_csv(SRC / f'evalx_patients_{setting}.csv')
    P = P[P.status == 'completed'].reset_index(drop=True)
    # timepoint-level table (GT lesion x timepoint that the pipeline segmented)
    rows = []
    for tag in ('bl', 'fu'):
        d = L.dropna(subset=[f'pred_vol_{tag}_mL', f'gt_vol_{tag}_mL']).copy()
        rows.append(pd.DataFrame({'patient_id': d.patient_id, 'dice': d[f'dice_{tag}'], 'iou': d[f'iou_{tag}'], 'hd95': d[f'hd95_mm_{tag}'],
                                  'gv': d[f'gt_vol_{tag}_mL'], 'pv': d[f'pred_vol_{tag}_mL'],
                                  'gd': d[f'gt_diam_{tag}_mm'], 'pd': d[f'pred_diam_{tag}_mm']}))
    T = pd.concat(rows, ignore_index=True)
    return L, T, P


def metrics(L, T, P):
    """All metrics for the given (already resampled) tables; pure numpy."""
    m = {}
    # --- Cut (segmentation) on segmented GT lesion-timepoints
    m['seg_dice_mean'] = np.nanmean(T.dice); m['seg_dice_median'] = np.nanmedian(T.dice)
    m['seg_iou_mean'] = np.nanmean(T.iou); m['seg_hd95_median_mm'] = np.nanmedian(T.hd95); m['seg_hd95_mean_mm'] = np.nanmean(T.hd95)
    m['seg_dice_gt_0.5_rate'] = np.nanmean(T.dice > .5)
    m['lesion_sensitivity_final'] = L.detected.mean()
    # --- Measure
    ve = (T.pv - T.gv).abs(); vr = (T.pv - T.gv) / T.gv
    m['volume_MAE_mL'] = ve.mean(); m['volume_median_abs_rel_err'] = vr.abs().median(); m['volume_median_signed_rel_bias'] = vr.median()
    ok = (T.pv > 0) & (T.gv > 0)
    m['volume_log_pearson_r'] = np.corrcoef(np.log(T.pv[ok]), np.log(T.gv[ok]))[0, 1] if ok.sum() > 2 else np.nan
    de = T.pd - T.gd
    m['diameter_MAE_mm'] = de.abs().mean(); m['diameter_median_abs_mm'] = de.abs().median(); m['diameter_bias_mm'] = de.mean()
    m['diameter_within_5mm_rate'] = (de.abs() <= 5).mean()
    # --- Match
    tp, pp, gp = P.correct_pairs.sum(), P.pred_pairs.sum(), P.gt_persistent.sum()
    m['match_precision'] = tp / pp if pp else np.nan; m['match_recall'] = tp / gp if gp else np.nan
    m['match_f1'] = 2 * tp / (pp + gp) if (pp + gp) else np.nan
    pc = L.pair_correct.dropna(); m['match_accuracy_given_both_segmented'] = pc.astype(float).mean() if len(pc) else np.nan
    # --- Status (3 classes, undetected counted as a fourth predicted category)
    conf = np.zeros((4, 4))
    t = L.gt_class.map({c: i for i, c in enumerate(CLASSES)}).values
    pcode = L.pred_status.map({c: i for i, c in enumerate(CLASSES)}).fillna(3).astype(int).values
    np.add.at(conf, (t.astype(int), pcode), 1)
    f1s = f1_from_conf(conf); w = conf[:3].sum(1)
    m['status_accuracy'] = np.trace(conf[:3, :3]) / conf.sum(); m['status_macro_f1'] = np.nanmean(f1s)
    m['status_weighted_f1'] = np.nansum(f1s * w) / w.sum(); m['status_cohens_kappa'] = kappa(conf)
    for i, c in enumerate(CLASSES):
        m[f'recall_{c}'] = conf[i, i] / conf[i].sum() if conf[i].sum() else np.nan
        m[f'precision_{c}'] = conf[i, i] / conf[:, i].sum() if conf[:, i].sum() else np.nan
    # --- Change (enlarge / shrink) on correctly matched persistent pairs
    ch = L.dropna(subset=['pred_change_pct', 'gt_change_pct'])
    if len(ch):
        m['change_direction_agreement'] = (np.sign(ch.pred_change_pct) == np.sign(ch.gt_change_pct)).mean()
        m['change_pct_MAE'] = (ch.pred_change_pct - ch.gt_change_pct).abs().mean()
        cc = np.zeros((3, 3))
        np.add.at(cc, (ch.gt_change_class.map({c: i for i, c in enumerate(CHG)}).values.astype(int),
                       ch.pred_change_class.map({c: i for i, c in enumerate(CHG)}).values.astype(int)), 1)
        m['change_class_accuracy'] = np.trace(cc) / cc.sum(); m['change_class_kappa'] = kappa(cc)
    else:
        for k in ('change_direction_agreement', 'change_pct_MAE', 'change_class_accuracy', 'change_class_kappa'):
            m[k] = np.nan
    # --- Patient level
    m['task_completion_rate'] = 1.0  # P holds completed patients only; failures are counted by the caller
    m['false_new_per_patient'] = P.false_new_lesions.mean(); m['false_disappeared_per_patient'] = P.false_disappeared_lesions.mean()
    m['new_count_MAE'] = (P.pred_new - P.gt_new).abs().mean(); m['disappeared_count_MAE'] = (P.pred_disappeared - P.gt_disappeared).abs().mean()
    m['gt_lesions_per_patient'] = (P.gt_bl + P.gt_fu).mean(); m['pred_lesions_per_patient'] = (P.pred_bl + P.pred_fu).mean()
    okb = P.gt_burden_bl_mL > 0
    m['burden_bl_median_abs_rel_err'] = ((P.pred_burden_bl_mL - P.gt_burden_bl_mL).abs() / P.gt_burden_bl_mL)[okb].median()
    g = ((P.gt_burden_fu_mL - P.gt_burden_bl_mL) / P.gt_burden_bl_mL * 100)[okb]; pr = ((P.pred_burden_fu_mL - P.pred_burden_bl_mL) / P.pred_burden_bl_mL * 100)[okb]
    m['burden_change_direction_agreement'] = (np.sign(g) == np.sign(pr)).mean() if okb.sum() else np.nan
    m['burden_change_pct_MAE'] = (g - pr).abs().mean() if okb.sum() else np.nan
    m['pipeline_seconds_per_patient'] = P.runtime_seconds.mean()
    return m


def group_index(df):
    return {k: v.values for k, v in df.groupby('patient_id').indices.items()} if False else {k: np.asarray(v) for k, v in df.groupby('patient_id').indices.items()}


def main():
    pairs = pd.DataFrame(json.loads((SRC / 'pairs.json').read_text()))
    cohort = dict(zip(pairs.patient_id, pairs.cohort))
    tabs = {s: load(s) for s in SETTINGS}
    completion = {}
    for s in SETTINGS:
        Pall = pd.read_csv(SRC / f'evalx_patients_{s}.csv')
        for c in ('locked', 'dev'):
            sub = Pall[Pall.cohort == c]
            completion[f'{s}/{c}'] = dict(attempted=int(len(sub)), completed=int((sub.status == 'completed').sum()),
                                          failed_ids=sub[sub.status != 'completed'].patient_id.tolist())
    # common completed locked patients (paired S0/A)
    done = set.intersection(*[set(P.patient_id) for _, _, P in tabs.values()])
    locked = sorted(p for p in done if cohort[p] == 'locked'); dev = sorted(p for p in done if cohort[p] == 'dev')
    print('locked completed', len(locked), 'dev completed', len(dev), completion)

    def sub(s, pids):
        L, T, P = tabs[s]; k = set(pids)
        return L[L.patient_id.isin(k)], T[T.patient_id.isin(k)], P[P.patient_id.isin(k)]

    point = {(s, c): metrics(*sub(s, pids)) for s in SETTINGS for c, pids in (('locked', locked), ('dev', dev))}
    metric_names = list(point[('A', 'locked')])
    # bootstrap (locked cohort): resample patients, take every row of the drawn patients
    idx = {}
    for s in SETTINGS:
        L, T, P = (x[x.patient_id.isin(locked)].reset_index(drop=True) for x in tabs[s])
        idx[s] = (L, T, P, group_index(L), group_index(T), group_index(P))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(locked), size=(NB, len(locked)))
    boot = []
    for b in range(NB):
        pids = [locked[i] for i in draws[b]]
        for s in SETTINGS:
            L, T, P, gL, gT, gP = idx[s]
            cat = lambda df, g: df.iloc[np.concatenate([g[p] for p in pids if p in g])]
            mm = metrics(cat(L, gL), cat(T, gT), cat(P, gP))
            boot.extend((b, s, k, v) for k, v in mm.items())
    boot = pd.DataFrame(boot, columns=['bootstrap_id', 'setting', 'metric', 'value'])
    boot.to_csv(DST / 'bootstrap_metrics_full.csv', index=False)
    wide = boot.pivot_table(index=['bootstrap_id', 'metric'], columns='setting', values='value').reset_index()
    rows = []
    for k in metric_names:
        w = wide[wide.metric == k]
        for s in SETTINGS:
            v = w[s].dropna()
            rows.append(dict(metric=k, contrast=s, estimate=point[(s, 'locked')][k], ci_lo=v.quantile(.025) if len(v) else np.nan,
                             ci_hi=v.quantile(.975) if len(v) else np.nan, n_boot_valid=len(v), dev_estimate=point[(s, 'dev')][k]))
        d = (w['A'] - w['S0']).dropna()
        rows.append(dict(metric=k, contrast='A-S0', estimate=point[('A', 'locked')][k] - point[('S0', 'locked')][k],
                         ci_lo=d.quantile(.025) if len(d) else np.nan, ci_hi=d.quantile(.975) if len(d) else np.nan, n_boot_valid=len(d),
                         dev_estimate=point[('A', 'dev')][k] - point[('S0', 'dev')][k]))
    st = pd.DataFrame(rows)
    st.to_csv(DST / 'exp22_full_statistics.csv', index=False)
    # per-patient long table for plotting
    long = []
    for s in SETTINGS:
        L, T, P = tabs[s]
        for pid in sorted(done):
            l, t, p = (x[x.patient_id == pid] for x in (L, T, P))
            for k, v in metrics(l, t, p).items():
                long.append((pid, cohort[pid], s, k, v))
    pd.DataFrame(long, columns=['patient_id', 'cohort', 'setting', 'metric', 'value']).to_csv(DST / 'patient_metrics_long_full.csv', index=False)
    (DST / 'exp22_full_meta.json').write_text(json.dumps(dict(n_locked=len(locked), n_dev=len(dev), completion=completion, n_bootstrap=NB, seed=SEED,
                                                              n_gt_lesions_locked={s: int(len(sub(s, locked)[0])) for s in SETTINGS}), indent=1))
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300)
    print(st[st.contrast.isin(['A', 'S0', 'A-S0'])].assign(v=lambda d: d.apply(lambda r: f'{r.estimate:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]', axis=1)).pivot(index='metric', columns='contrast', values='v').to_string())


if __name__ == '__main__':
    main()
