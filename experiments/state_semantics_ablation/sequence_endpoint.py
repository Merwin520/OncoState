#!/usr/bin/env python3
"""Sequence-level empirical endpoint for the state-semantics ablation (I-SPY2, all 719 patients).

Every variant processes the SAME per-patient event stream; after every event the SAME reader code answers five standard questions from
the variant's STORED state (what a later session would see), and each answer is scored against an oracle built from the event log.
  events     scheduled visits T0..T3 (real data, natural missing visits included); an observed intermediate visit delivered late
             (after the next scheduled visit); a verified correction of an earlier visit's FTV (+1.0 cc marker); a duplicate delivery
             of an already synchronized visit; a silent storage-level overwrite of a stored historical value (fault outside the transition
             operator). Event occurrence per patient is Bernoulli with the stated rates (fixed seed); values and visit availability are real.
  questions  trajectory  - FTV value of every received visit (corrections applied)
             missing     - which visits that were due have not been received
             sync_point  - which protocol visit the state is synchronized to
             source      - the source series UID of every received FTV value
             integrity   - has stored history been altered outside logged revisions
Endpoint: per-patient state error rate = wrong answers / answers over all checkpoints (and per question). Paired patient bootstrap
Full - variant (Exp 2.8 procedure). Event rates are assumptions; scenarios 0.5x / 1x / 2x are reported as sensitivity.
"""
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variants as V  # noqa: E402
from evaluation.statistics import paired_bootstrap, patient_bootstrap  # noqa: E402
from features.measurement_features import ISPY2EvidenceSource, TP2, ispy2_cfg  # noqa: E402
from run_ablation import STORES, Backend, fval  # noqa: E402

OUT = V.REPO / 'outputs/state_semantics_ablation'
RUN = OUT / 'runs_sequence'
BASE_RATES = dict(late=.25, correction=.25, duplicate=.10, overwrite=.05)
SCENARIOS = {'0.5x': .5, '1x': 1.0, '2x': 2.0}
SEED = 20261006
QUESTIONS = ['trajectory', 'missing', 'sync_point', 'source', 'integrity']


def make_stream(source, pid, rates, rng):
    """List of events (kind, timepoint) for one patient; uses only which visits exist in the real data."""
    obs = {tp: source.observation(pid, tp) for tp in TP2}
    present = [tp for tp in TP2 if obs[tp] is not None]
    late_tp = None
    cand = [tp for tp in ('T1', 'T2') if obs[tp] is not None]
    if cand and rng.random() < rates['late']:
        late_tp = cand[rng.integers(len(cand))]
    ev = []
    for tp in TP2:
        ev.append(('scheduled', tp, None if tp == late_tp or obs[tp] is None else tp))   # (kind, due timepoint, delivered timepoint or None)
        if late_tp and TP2.index(tp) == TP2.index(late_tp) + 1:
            ev.append(('late', late_tp, late_tp))
    received = lambda i: [e[2] for e in ev[:i + 1] if e[2]]  # noqa: E731
    # correction: after a random scheduled event, of a visit received by then
    if present and rng.random() < rates['correction']:
        idx = [i for i, e in enumerate(ev) if received(i)]
        i = idx[rng.integers(len(idx))]; r = received(i)
        ev.insert(i + 1, ('correction', None, r[rng.integers(len(r))]))
    if present and rng.random() < rates['duplicate']:
        idx = [i for i, e in enumerate(ev) if received(i)]
        i = idx[rng.integers(len(idx))]; r = received(i)
        ev.insert(i + 1, ('duplicate', None, r[rng.integers(len(r))]))
    if present and rng.random() < rates['overwrite']:
        idx = [i for i, e in enumerate(ev) if any(fval({'history': {t: obs[t]}}, t) is not None for t in received(i))]
        if idx:
            i = idx[rng.integers(len(idx))]
            r = [t for t in received(i) if fval({'history': {t: obs[t]}}, t) is not None]
            ev.insert(i + 1, ('overwrite', None, r[rng.integers(len(r))]))
    return ev, obs


def read(state, store, pid, versioned):
    """The same reader for every variant: answers from the stored state only."""
    st = state or {'history': {}, 'observation_status': {}, 'current_timepoint': None, 'protocol_timepoints': list(TP2)}
    order = list(TP2)
    traj = {tp: fval(st, tp) for tp in st['history']}
    status = st.get('observation_status') or {}
    if status:
        missing = sorted(tp for tp, s in status.items() if s in ('not_observed', 'hidden_stress_test'))
    else:                                                               # implicit: infer from protocol order and the latest visit
        cur = st.get('current_timepoint')
        missing = sorted(tp for tp in order if cur in order and order.index(tp) <= order.index(cur) and tp not in st['history'])
    src = {}
    for tp, rec in st['history'].items():
        p = (rec.get('fields', {}).get('ftv_pe_cc') or {}).get('provenance') or {}
        src[tp] = p.get('source_series_uid')
    v = store.verify(pid) if versioned else None
    altered = (v is False) if versioned and (store.lineage(pid)['versions']) else False
    return dict(trajectory=traj, missing=missing, sync_point=st.get('current_timepoint'), source=src, integrity=altered)


def same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 1e-9
    return a == b


def tamper(b, pid, tp):
    st = b.store
    if b.versioned:
        lin = st.lineage(pid)['versions']
        f = st.pdir(pid) / lin[-1]['file']
    else:
        f = st._f(pid)
    s = json.loads(f.read_text())
    if tp in s['history'] and fval(s, tp) is not None:
        s['history'][tp]['fields']['ftv_pe_cc']['value'] = fval(s, tp) * 1.5 + 1.0
        f.write_text(json.dumps(s, indent=1, sort_keys=True, default=str))
        return fval(s, tp)
    return None


def run_patient(b, source, pid, ev, obs):
    cfg = ispy2_cfg(pid)
    truth = dict(values={}, received=[], due=[], overwritten=False, uid={})
    rows = []
    for k, (kind, due, tp) in enumerate(ev):
        if kind == 'scheduled':
            truth['due'].append(due)
            o = obs[due] if tp else None
            try:
                b.step(pid, due, o, source, cfg, now=f'seq-{k}')
            except (V.HistoricalWriteRefused, ValueError):
                pass
            if tp:
                truth['received'].append(tp); truth['values'][tp] = fval({'history': {tp: o}}, tp); truth['uid'][tp] = o['series_uids'][0]
        elif kind == 'late':
            b.step(pid, tp, obs[tp], source, cfg, event_type='late_arrival', now=f'seq-{k}')
            truth['received'].append(tp); truth['values'][tp] = fval({'history': {tp: obs[tp]}}, tp); truth['uid'][tp] = obs[tp]['series_uids'][0]
        elif kind == 'correction':
            if truth['values'].get(tp) is None:
                continue
            new = truth['values'][tp] + 1.0
            b.correct(pid, tp, 'ftv_pe_cc', new, cfg, 'verified correction (sequence endpoint)')
            truth['values'][tp] = new
        elif kind == 'duplicate':
            try:
                b.step(pid, tp, obs[tp], source, cfg, event_type='late_arrival', now=f'seq-{k}')
            except (V.HistoricalWriteRefused, ValueError):
                pass
        elif kind == 'overwrite':
            if tamper(b, pid, tp) is not None:
                truth['overwritten'] = True
        state = b.store.latest(pid)
        ans = read(state, b.store, pid, b.versioned)
        ref = dict(trajectory={t: truth['values'][t] for t in truth['received']},
                   missing=sorted(t for t in truth['due'] if t not in truth['received']),
                   sync_point=truth['due'][-1] if truth['due'] else None,
                   source={t: truth['uid'][t] for t in truth['received']}, integrity=truth['overwritten'])
        if truth['overwritten']:                    # after a storage fault the stored values are wrong for every variant; the
            ref['trajectory'] = None                # value question is not scored then (the integrity question is)
        for q in QUESTIONS:
            if ref[q] is None and q == 'trajectory':
                continue
            rows.append(dict(event_index=k, event=kind, question=q, correct=bool(same(ref[q], ans[q]))))
    return rows


def main():
    t0 = time.time()
    if RUN.exists():
        shutil.rmtree(RUN)
    source = ISPY2EvidenceSource()
    patients = sorted(source.visits.patient_id.unique())
    per, events_log = [], []
    for scen, mult in SCENARIOS.items():
        rates = {k: min(1.0, v * mult) for k, v in BASE_RATES.items()}
        rng = np.random.default_rng(SEED)
        streams = {p: make_stream(source, p, rates, rng) for p in patients}
        for p, (ev, _) in streams.items():
            events_log += [dict(scenario=scen, patient_id=p, event_index=i, event=e[0], due=e[1], timepoint=e[2]) for i, e in enumerate(ev)]
        for v in V.VARIANTS:
            b = Backend(v, RUN / scen / v)
            for p in patients:
                ev, obs = streams[p]
                rows = run_patient(b, source, p, ev, obs)
                d = pd.DataFrame(rows)
                r = dict(scenario=scen, variant=v, patient_id=p, n_answers=len(d), state_error_rate=1 - d.correct.mean(),
                         any_error=float((~d.correct).any()))
                for q in QUESTIONS:
                    x = d[d.question == q]
                    r[f'error_{q}'] = 1 - x.correct.mean() if len(x) else np.nan
                for kind in ('late', 'correction', 'duplicate', 'overwrite'):
                    r[f'has_{kind}'] = any(e[0] == kind for e in ev)
                per.append(r)
            print(f'[{time.time() - t0:.0f}s] {scen} {v} done', flush=True)
    per = pd.DataFrame(per); per.to_csv(OUT / 'sequence_per_patient.csv', index=False)
    pd.DataFrame(events_log).to_csv(OUT / 'sequence_event_streams.csv', index=False)
    agg, pair = [], []
    eps = ['state_error_rate', 'any_error'] + [f'error_{q}' for q in QUESTIONS]
    for scen, d in per.groupby('scenario'):
        full = d[d.variant == 'A_full'].set_index('patient_id')
        for v in V.VARIANTS:
            x = d[d.variant == v].set_index('patient_id')
            for ep in eps:
                bt = patient_bootstrap(x[ep].values)
                agg.append(dict(scenario=scen, variant=v, endpoint=ep, estimate=bt['estimate'], ci95_lo=bt['ci_low'], ci95_hi=bt['ci_high'],
                                n_patients=bt['n_patients']))
                if v != 'A_full':
                    j = pd.concat([full[ep], x[ep]], axis=1, keys=['a', 'b']).dropna()
                    pb = paired_bootstrap(j.b.values, j.a.values)          # variant - Full (positive = more errors without the mechanism)
                    pair.append(dict(scenario=scen, endpoint=ep, comparison=f'{v} - A_full', difference=pb['estimate'], ci95_lo=pb['ci_low'],
                                     ci95_hi=pb['ci_high'], n_patients=pb['n_patients'], wilcoxon_p=pb.get('wilcoxon_p')))
    pd.DataFrame(agg).to_csv(OUT / 'sequence_aggregate.csv', index=False)
    pd.DataFrame(pair).to_csv(OUT / 'sequence_paired_bootstrap.csv', index=False)
    (OUT / 'sequence_config.json').write_text(json.dumps(dict(base_rates=BASE_RATES, scenarios=SCENARIOS, seed=SEED, questions=QUESTIONS,
                                                             n_patients=len(patients), note=__doc__), indent=1))
    a = pd.DataFrame(agg)
    print(a[a.endpoint.isin(['state_error_rate', 'any_error'])].pivot_table(index=['scenario', 'endpoint'], columns='variant', values='estimate').round(4))
    print(f'finished in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
