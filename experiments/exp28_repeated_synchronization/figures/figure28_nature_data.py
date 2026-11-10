"""Source data of the I-SPY synchronization analysis: write figures/figure28_nature_data.csv (long format, every plotted element).

Re-uses figure28_data.build() for the summary panels and adds the dense series:
  raster      : visit availability, 719 patients x 4 visits (sorted by visit pattern, then baseline FTV)
  traj        : FTV held in each patient's final state at every visit (NaN = explicitly missing; never interpolated)
  traj_summary: median and IQR of FTV per visit
  lineage     : one stress-test patient's five versions (T1 hidden -> late arrival), deterministic choice
  i1_traj     : I-SPY1 MRI longest diameter held in the replication states
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.figure28_data import build as build_summary  # noqa: E402
from paths import EXPORTS, PKG, STATES, work_dir  # noqa: E402
from state.versioning import VersionStore  # noqa: E402

TP, V1 = ['T0', 'T1', 'T2', 'T3'], ['V0', 'V1', 'V2', 'V3']
OUT = PKG / 'figures' / 'figure28_nature_data.csv'


def main():
    base = build_summary()
    R = []

    def r(panel, series='', label='', patient_id='', x=np.nan, y=np.nan, lo=np.nan, hi=np.nan, value=np.nan, n=np.nan, note=''):
        R.append(dict(panel=panel, series=series, label=label, patient_id=patient_id, x=x, y=y, lo=lo, hi=hi, value=value, n=n, note=note))

    store = VersionStore(STATES)
    pm = pd.read_csv(EXPORTS / 'ispy2_patient_manifest.csv', dtype={'patient_id': str})
    finals = {p: store.latest(p) for p in pm.patient_id}
    ftv = {(p, t): (f['history'].get(t, {}).get('fields', {}).get('ftv_pe_cc', {}).get('value')) for p, f in finals.items() for t in TP}

    # raster: sort by pattern frequency, then baseline FTV (descending)
    freq = pm.visit_pattern.value_counts()
    pm['freq'] = pm.visit_pattern.map(freq)
    pm['bl'] = [ftv[(p, 'T0')] if ftv[(p, 'T0')] is not None else -1 for p in pm.patient_id]
    order = pm.sort_values(['freq', 'visit_pattern', 'bl'], ascending=[False, True, False]).reset_index(drop=True)
    for i, row in order.iterrows():
        for j, t in enumerate(TP):
            r('raster', 'raster', t, row.patient_id, x=j, y=i, value=int(row.visit_pattern[j] != '-'), note=row.visit_pattern)

    # trajectories held in the state
    for p in pm.patient_id:
        for j, t in enumerate(TP):
            v = ftv[(p, t)]
            r('traj', 'traj', t, p, x=j, y=np.nan if v is None else v, value=int(v is not None),
              note=finals[p]['observation_status'].get(t, ''))
    for j, t in enumerate(TP):
        v = np.array([ftv[(p, t)] for p in pm.patient_id if ftv[(p, t)] is not None], float)
        r('traj', 'traj_summary', t, x=j, y=float(np.median(v)), lo=float(np.percentile(v, 25)), hi=float(np.percentile(v, 75)), n=len(v),
          value=int(sum(ftv[(p, t)] is None for p in pm.patient_id)), note='value = patients with this visit explicitly missing')

    # one stress-test lineage (deterministic: first patient by ID in the stress table)
    st = pd.read_csv(EXPORTS / 'missingness_stress.csv', dtype={'patient_id': str})
    sp = sorted(st.patient_id)[0]
    ss = VersionStore(work_dir() / 'stress_states_missing')
    for k, (s, e) in enumerate(zip(ss.all_versions(sp), ss.lineage(sp)['versions'])):
        for j, t in enumerate(TP):
            status = s['observation_status'].get(t, 'pending')
            r('lineage', 'lineage', t, sp, x=j, y=k, note=f"{s['state_version']}|{e['event_type']}|{status}",
              value={'observed': 1, 'late_arrival': 2, 'hidden_stress_test': 3, 'not_observed': 4}.get(status, 0))

    # I-SPY1 trajectories (longest diameter, mm) from the replication states
    i1 = VersionStore(work_dir() / 'ispy1_states')
    for pdir in sorted((work_dir() / 'ispy1_states').iterdir()):
        if not pdir.is_dir():
            continue
        f = i1.latest(pdir.name)
        for j, t in enumerate(V1):
            v = f['history'].get(t, {}).get('fields', {}).get('longest_diameter_mm', {}).get('value')
            r('i1_traj', 'i1_traj', t, pdir.name, x=j, y=np.nan if v is None else v, value=int(v is not None))

    out = pd.concat([base, pd.DataFrame(R)], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(OUT.name, out.groupby('panel').size().to_dict(), 'stress example', sp)


if __name__ == '__main__':
    main()
