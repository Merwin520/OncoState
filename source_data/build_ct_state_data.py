#!/usr/bin/env python3
"""Source data: CT lesion state, segmentation and lesion matching. Reads source_data/ct_figure_inputs (build_ct_figure_inputs.py).

    python build_ct_state_data.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / 'ct_figure_inputs'   # written by build_ct_figure_inputs.py from the regenerated pipeline
BINS, BIN_NAMES = [0, .5, 2, 10, np.inf], ['<0.5', '0.5–2', '2–10', '>10']


def build() -> pd.DataFrame:
    R = []

    def r(panel, series='', label='', x=np.nan, y=np.nan, lo=np.nan, hi=np.nan, value=np.nan, n=np.nan, note=''):
        R.append(dict(panel=panel, series=series, label=label, x=x, y=y, lo=lo, hi=hi, value=value, n=n, note=note))

    stats = pd.read_csv(SRC / 'stats.csv')
    meta = json.loads((SRC / 'meta.json').read_text())
    diag = json.loads((SRC / 'registration_diagnosis.json').read_text())

    def est(metric, s):
        q = stats[(stats.metric == metric) & (stats.contrast == s)].iloc[0]
        return float(q.estimate), float(q.ci_lo), float(q.ci_hi)

    r('a', 'stat', 'n_locked', value=meta['n_locked']); r('a', 'stat', 'n_gt_lesions', value=meta['n_gt_lesions_locked']['A'])
    la = pd.read_csv(SRC / 'lesions_A.csv')
    rows = []
    for tag in ('bl', 'fu'):
        d = la.dropna(subset=[f'pred_vol_{tag}_mL', f'gt_vol_{tag}_mL'])
        rows.append(pd.DataFrame({'dice': d[f'dice_{tag}'], 'gv': d[f'gt_vol_{tag}_mL'], 'pv': d[f'pred_vol_{tag}_mL'],
                                  'gd': d[f'gt_diam_{tag}_mm'], 'pd': d[f'pred_diam_{tag}_mm']}))
    ta = pd.concat(rows, ignore_index=True)
    ta['bin'] = pd.cut(ta.gv, BINS, labels=BIN_NAMES, right=False)
    for b, g in ta.dropna(subset=['dice']).groupby('bin', observed=True):
        q = g.dice.quantile([.05, .25, .5, .75, .95]).to_numpy()
        r('b1', 'box', b, value=q[2], lo=q[1], hi=q[3], x=q[0], y=q[4], n=len(g), note='dice median, IQR (lo/hi), 5th/95th pct (x/y)')
        for v in g.dice[(g.dice < q[0]) | (g.dice > q[4])]:
            r('b1', 'outlier', b, value=float(v))
    ok = (ta.pv > 0) & (ta.gv > 0)
    for g, p in zip(ta.gv[ok], ta.pv[ok]):
        r('b2', 'point', '', x=float(g), y=float(p))
    for k in ('volume_log_pearson_r', 'volume_median_abs_rel_err', 'volume_median_signed_rel_bias', 'diameter_bias_mm'):
        e = est(k, 'A'); r('b2', 'stat', k, value=e[0], lo=e[1], hi=e[2])
    dd = ta.dropna(subset=['gd', 'pd'])
    diff = dd.pd - dd.gd
    r('b2', 'stat', 'diameter_LoA', lo=float(diff.mean() - 1.96 * diff.std()), hi=float(diff.mean() + 1.96 * diff.std()))
    for m_, d_ in zip((dd.gd + dd.pd) / 2, diff):
        r('b2', 'ba', '', x=float(m_), y=float(d_))
    for m_ in ('match_f1', 'status_weighted_f1', 'status_macro_f1', 'status_cohens_kappa', 'change_class_accuracy'):
        for s in ('S0', 'A'):
            if m_ == 'change_class_accuracy' and s == 'S0':
                continue                                       # trivially 1.0 with reference masks
            e = est(m_, s); r('c', s, m_, value=e[0], lo=e[1], hi=e[2], n=274)
    for c in ('persistent', 'disappearing', 'newly_appearing'):
        for kind in ('recall', 'precision'):
            for s in ('S0', 'A'):
                e = est(f'{kind}_{c}', s); r('d1', s, f'{kind}_{c}', value=e[0], lo=e[1], hi=e[2])
    ch = la.dropna(subset=['pred_change_pct', 'gt_change_pct'])
    for _, q in ch.iterrows():
        r('d2', 'same' if q.gt_change_class == q.pred_change_class else 'differs', '', x=float(q.gt_change_pct), y=float(q.pred_change_pct))
    for k in ('change_class_accuracy', 'change_class_kappa', 'change_direction_agreement', 'burden_change_direction_agreement'):
        e = est(k, 'A'); r('d2', 'stat', k, value=e[0], lo=e[1], hi=e[2])
    reg = pd.read_csv(SRC / 'regsanity.csv')
    for v in reg.dist_shared_id_mm.dropna():
        r('e', 'matched', '', x=float(v))
    for v in reg.dist_nearest_other_fu_lesion_mm.dropna():
        r('e', 'unmatched', '', x=float(v))
    mf = pd.read_csv(SRC / 'match_f1_S0.csv').set_index('patient_id').iloc[:, 0]
    pdist = reg.groupby('patient_id').dist_shared_id_mm.median()
    jj = pd.concat([pdist, mf], axis=1, keys=['dist', 'f1']).dropna()
    for pid, q in jj.iterrows():
        r('e', 'patient', pid, x=float(q.dist), y=float(q.f1))
    r('e', 'stat', 'n_bad', value=diag['n_bad'], note='patients with reference persistent lesions but match F1 = 0 (S0)')
    return pd.DataFrame(R)


def main():
    out = Path(__file__).resolve().parent / 'out' / 'figure22_nature_data.csv'
    out.parent.mkdir(exist_ok=True)
    d = build()
    d.to_csv(out, index=False)
    print(f'wrote {out} ({len(d)} rows)')


if __name__ == '__main__':
    main()
