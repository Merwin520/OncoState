#!/usr/bin/env python3
"""LUMIERE scan table for Experiment A (no outcome is computed here).

One row per rated scan: patient, time point, week, order, role (preop / postop / followup), expert RANO rating, 12-week flag, and the
enhancing (CE) and non-enhancing T2/FLAIR (T2) volumes in mL from the two independent automated pipelines:
  HD-GLIO-AUTO   CE = 'Contrast-enhancing', T2 = 'Non-enhancing'
  DeepBraTumIA   CE = 'Contrast-enhancing', T2 = 'Edema'
A label with no pyradiomics row in a scan whose segmentation exists (data-completeness table) has volume 0 (pyradiomics skips empty
labels); a scan without the segmentation has no measurement (NaN).
    python data.py  -> results/scans.csv
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'LUMIERE/tables'))
PIPE = {'hd': ('hdglioauto', 'HD-GLIO-AUTO', {'Contrast-enhancing': 'ce', 'Non-enhancing': 't2'}),
        'db': ('deepbratumia', 'DeepBraTumIA', {'Contrast-enhancing': 'ce', 'Edema': 't2'})}


def week(tp):
    m = re.match(r'week-(\d+)(?:-(\d+))?', tp)
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 10


def volumes(key):
    f, col, lab = PIPE[key]
    x = pd.read_csv(T / f'LUMIERE-pyradiomics-{f}-features.csv', usecols=['Patient', 'Time point', 'Label name', 'Sequence',
                                                                           'original_shape_VoxelVolume'])
    x = x[(x.Sequence == 'CT1') & x['Label name'].isin(lab)]
    v = x.pivot_table(index=['Patient', 'Time point'], columns='Label name', values='original_shape_VoxelVolume', aggfunc='first')
    v = v.rename(columns=lab) / 1000.0
    comp = pd.read_csv(T / 'LUMIERE-datacompleteness.csv')
    have = {(p, t) for p, t, h in zip(comp.Patient, comp.Timepoint, comp[col]) if str(h).strip().lower() == 'x'}
    return v, have


def main():
    r = pd.read_csv(T / 'LUMIERE-ExpertRating-v202211.csv')
    r.columns = ['patient', 'tp', 'lt3m', 'nonmeas', 'rating', 'rationale']
    r['rating'] = r.rating.astype(str).str.strip()
    r['lt3m'] = r.lt3m.notna().astype(int); r['nonmeas'] = r.nonmeas.notna().astype(int)
    r['week'] = r.tp.map(week)
    # duplicated ratings of one scan in the source table: identical -> one row; conflicting with a Post-Op rating -> post-operative
    # baseline; other conflicts -> 'conflict' (not evaluated; its measurement may serve as a prior). Data-cleaning rule, see REPORT.
    keep = []
    for (p_, t_), g in r.groupby(['patient', 'tp'], sort=False):
        row = g.iloc[0].copy()
        if g.rating.nunique() > 1:
            row['rating'] = 'Post-Op' if g.rating.str.startswith('Post-Op').any() else 'conflict'
        keep.append(row)
    r = pd.DataFrame(keep)
    r = r.sort_values(['patient', 'week']).reset_index(drop=True)
    r['role'] = np.select([r.rating == 'Pre-Op', r.rating.str.startswith('Post-Op'), r.rating.isin(['PD', 'SD', 'PR', 'CR'])],
                          ['preop', 'postop', 'followup'], np.where(r.rating == 'conflict', 'conflict', 'unrated'))
    for k in PIPE:
        v, have = volumes(k)
        for c in ('ce', 't2'):
            vals = []
            for p, t in zip(r.patient, r.tp):
                if (p, t) not in have:
                    vals.append(np.nan)
                else:
                    x = v[c].get((p, t), np.nan) if c in v else np.nan
                    vals.append(0.0 if pd.isna(x) else float(x))
            r[f'{k}_{c}'] = vals
    # baseline = the last post-operative scan with an HD-GLIO measurement; follow-ups after it are evaluated
    rows = []
    for p, g in r.groupby('patient', sort=True):
        g = g.copy()
        post = g[(g.role == 'postop') & g.hd_ce.notna()]
        bw = post.week.max() if len(post) else np.nan
        g['baseline'] = (g.week == bw) & (g.role == 'postop')
        g['evaluable'] = (g.role == 'followup') & (g.week > bw) & g.hd_ce.notna() & g.hd_t2.notna()
        g['prior_ok'] = (g.week >= bw) & g.hd_ce.notna() & g.hd_t2.notna() & g.role.isin(['postop', 'followup', 'conflict'])
        rows.append(g)
    out = pd.concat(rows)
    (HERE / 'results').mkdir(exist_ok=True)
    out.to_csv(HERE / 'results/scans.csv', index=False)
    print('scans', len(out), 'patients', out.patient.nunique(), 'evaluable follow-ups', int(out.evaluable.sum()),
          'patients with >=1 evaluable', out[out.evaluable].patient.nunique(), 'lt3m among evaluable', int(out[out.evaluable].lt3m.sum()),
          'both pipelines', int((out.evaluable & out.db_ce.notna()).sum()))


if __name__ == '__main__':
    main()
