"""Shared data layer of longitudinal-forecasting (pre-specified analysis plan): cohorts, scenario event streams, visit tables for B/H/S/M and the shared
feature function F. Official Exp 2.8 code (synchronize, revise, evidence sources) is used unchanged."""
import os
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E28 = REPO / 'experiments/exp28_repeated_synchronization'
sys.path.insert(0, str(E28))
from features.measurement_features import ISPY1EvidenceSource, ISPY2EvidenceSource, ispy1_cfg, ispy2_cfg  # noqa: E402
from state.synchronization import revise, synchronize  # noqa: E402

OUT = REPO / 'outputs/longitudinal_forecasting'
CLIN2 = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'exp28_work/metadata/source/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx'))
T2, T1 = ['T0', 'T1', 'T2', 'T3'], ['V0', 'V1', 'V2', 'V3']
M2 = [('ftv_pe_cc', 'ftv_pe', True), ('ftv_ser_cc', 'ftv_ser', True), ('longest_diameter_cm', 'ld', True),
      ('sphericity', 'sph', False), ('bpe_5slice_mean', 'bpe', False)]
FACTOR = 1.6
SCEN = {'natural': [], 'miss_T1': ['T1'], 'miss_T2': ['T2'], 'miss_T1T2': ['T1', 'T2'], 'revision': []}


def _f(obs, key):
    x = ((obs or {}).get('fields', {}).get(key) or {})
    return x.get('value') if x.get('available', x.get('value') is not None) else None


# ------------------------------------------------------------------------------------------------- I-SPY2
class ISPY2:
    def __init__(self):
        self.src = ISPY2EvidenceSource()
        c = pd.read_excel(CLIN2); c['pid'] = c.Patient_ID.astype(str); self.clin = c.set_index('pid')
        self.pts = sorted(self.src.visits.patient_id.unique())

    def record(self, pid, tp, scen, k):
        """Source record of visit tp as available at landmark k under scenario (None = no MRI acquired)."""
        if tp in SCEN[scen]:
            return None
        o = self.src.observation(pid, tp)
        if o is None:
            return None
        o = copy.deepcopy(o)
        if scen == 'revision' and tp == 'T1' and k < 2:            # wrong T1 record until the revision arrives with T2
            for key in ('ftv_pe_cc', 'ftv_ser_cc'):
                if o['fields'][key]['value'] is not None:
                    o['fields'][key]['value'] = o['fields'][key]['value'] * FACTOR
        return o

    def clinical(self, pid):
        r = self.src.clinical(pid)['fields']
        return dict(HR=r.get('HR'), HER2=r.get('HER2'), MP=r.get('MP'), age=r.get('age_at_screening'),
                    postmeno=float(str(r.get('menopausal_status', '')).startswith('Post')), arm=r.get('arm'))

    def table_from_source(self, pid, scen, k):
        """H: rebuilt from scratch from the source records that exist up to landmark k."""
        tab = {}
        for tp in T2[:k + 1]:
            o = self.record(pid, tp, scen, k)
            tab[tp] = None if o is None else dict(day=o.get('study_day_offset'), **{n: _f(o, key) for key, n, _ in M2})
        return tab

    def states(self, pid, scen, durable=True):
        """S: official synchronize (and revise) applied event by event; returns twin version after each landmark."""
        cfg = ispy2_cfg(pid); s, out = None, []
        for k, tp in enumerate(T2):
            if scen == 'revision' and k == 2 and s is not None and s['observation_status'].get('T1') == 'observed':
                corr = self.src.observation(pid, 'T1')
                if durable:
                    for key in ('ftv_pe_cc', 'ftv_ser_cc'):
                        s, _ = revise(s, 'T1', key, corr['fields'][key]['value'], reason='revised record', actor='radiology',
                                      source='revised T1 record', cfg=cfg, now='revision')
            s, _ = synchronize(s, tp, self.record(pid, tp, scen, k), clinical=self.src.clinical(pid) if s is None else None, cfg=cfg,
                               now=f'{scen}-{tp}')
            out.append(copy.deepcopy(s))
        return out

    @staticmethod
    def table_from_state(s, k):
        tab = {}
        for tp in T2[:k + 1]:
            rec = s['history'].get(tp)
            ok = s['observation_status'].get(tp) in ('observed', 'late_arrival') and rec is not None
            tab[tp] = dict(day=rec.get('study_day_offset'), **{n: _f(rec, key) for key, n, _ in M2}) if ok else None
        return tab

    def events(self, pid, scen):
        """Clinical events per landmark, as given to the memory agent (and implied for H and S)."""
        ev = []
        c = self.clinical(pid); c.pop('arm')
        for k, tp in enumerate(T2):
            e = []
            if k == 0:
                e.append(dict(type='clinical_record', content=c))
            if scen == 'revision' and k == 2 and self.record(pid, 'T1', 'natural', 0) is not None:
                o = self.record(pid, 'T1', 'natural', 2)
                e.append(dict(type='revised_record', note='the earlier T1 record contained wrong FTV values; this corrected record replaces it',
                              visit='T1', record=vrec(o)))
            o = self.record(pid, tp, scen, k)
            e.append(dict(type='visit_record', visit=tp, record=vrec(o)) if o is not None else
                     dict(type='missing_visit', visit=tp, note='no MRI was acquired at this scheduled visit'))
            ev.append(e)
        return ev


def vrec(o):
    return dict(study_day=o.get('study_day_offset'), **{key: _f(o, key) for key, _, _ in M2})


# ------------------------------------------------------------------------------------------------- I-SPY1 (external)
class ISPY1:
    def __init__(self):
        self.src = ISPY1EvidenceSource(); d = self.src.data
        self.all = list(d.index)
        self.pts = [p for p in d.index if not pd.isna(d.at[p, 'PCR']) and not pd.isna(d.at[p, 'MRI LD Baseline'])]

    def y(self, pid):
        return int(self.src.data.at[pid, 'PCR'])

    def clinical(self, pid):
        d = self.src.data
        return dict(HR=d.at[pid, 'HR Pos'], HER2=d.at[pid, 'Her2MostPos'], age=d.at[pid, 'age'])

    def table_from_source(self, pid, k):
        tab = {}
        for j, tp in enumerate(T1[:k + 1]):
            o = self.src.observation(pid, tp)
            tab[T2[j]] = None if o is None else dict(day=None, ld=o['fields']['longest_diameter_mm']['value'] / 10)
        return tab

    def states(self, pid):
        cfg = ispy1_cfg(pid); s, out = None, []
        for tp in T1:
            s, _ = synchronize(s, tp, self.src.observation(pid, tp), clinical=self.src.clinical(pid) if s is None else None, cfg=cfg, now=tp)
            out.append(copy.deepcopy(s))
        return out

    @staticmethod
    def table_from_state(s, k):
        tab = {}
        for j, tp in enumerate(T1[:k + 1]):
            rec = s['history'].get(tp)
            ok = s['observation_status'].get(tp) == 'observed' and rec is not None
            tab[T2[j]] = dict(day=None, ld=_f(rec, 'longest_diameter_mm') / 10) if ok else None
        return tab

    def events(self, pid):
        ev = []
        for k, tp in enumerate(T1):
            e = [dict(type='clinical_record', content=self.clinical(pid))] if k == 0 else []
            o = self.src.observation(pid, tp)
            e.append(dict(type='visit_record', visit=T2[k], record=dict(longest_diameter_cm=o['fields']['longest_diameter_mm']['value'] / 10))
                     if o is not None else dict(type='missing_visit', visit=T2[k], note='no MRI was acquired at this scheduled visit'))
            ev.append(e)
        return ev


# ------------------------------------------------------------------------------------------------- shared feature function F
def table_from_memory(mem, k, names):
    tab = {tp: None for tp in T2[:k + 1]}
    for v in (mem or {}).get('visits', []) or []:
        tp = str(v.get('visit', '')).strip().upper()
        if tp in tab and str(v.get('status', 'observed')).lower() == 'observed':
            num = lambda x: float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None  # noqa: E731
            vals = {n: num(v.get(key)) for key, n, _ in M2 if n in names}
            vals['day'] = num(v.get('study_day'))
            # observed = a visit record exists (any measurement), as in table_from_source / the state
            tab[tp] = vals if any(num(v.get(key)) is not None for key, _, _ in M2) else None
    return tab


def F(tab, k, clin, names=('ftv_pe', 'ftv_ser', 'ld', 'sph', 'bpe'), with_flags=True, summaries=True):
    LOG = {'ftv_pe', 'ftv_ser', 'ld'}
    nan = np.nan
    def val(tp, n):
        v = (tab.get(tp) or {}).get(n)
        if v is None:
            return nan
        return float(np.log(v + .1)) if n in LOG else float(v)
    out = {}
    key = names[0]
    for j, tp in enumerate(T2[:k + 1]):
        for n in names:
            out[f'{tp}_{n}'] = val(tp, n)
        if j:
            out[f'{tp}_ratio'] = val(tp, key) - val('T0', key)
        if with_flags:
            out[f'{tp}_observed'] = float(tab.get(tp) is not None)
    if summaries:
        seq = [(tp, val(tp, key), (tab.get(tp) or {}).get('day')) for tp in T2[:k + 1] if tab.get(tp) is not None and not np.isnan(val(tp, key))]
        out['n_observed'] = float(len(seq))
        out['cur'] = seq[-1][1] if seq else nan
        out['cur_vs_first'] = seq[-1][1] - seq[0][1] if len(seq) > 1 else 0.
        out['cur_vs_prev'] = seq[-1][1] - seq[-2][1] if len(seq) > 1 else 0.
        out['day_latest'] = float(seq[-1][2]) if seq and seq[-1][2] is not None else nan
    out.update({c: (nan if v is None else float(v)) for c, v in clin.items() if c != 'arm'})
    return out
