#!/usr/bin/env python3
"""State-semantics ablation on the frozen I-SPY2 Exp 2.8 cohort (variants A, B, D, E, F; C is documented as not separable).

Every variant runs the same four protocols with the same evidence source; between steps the state is always re-loaded from the variant's own
store (what a later session would see). Endpoints are computed from the returned/stored states, never typed in.
  P1 natural      all 719 patients, scheduled T0..T3 (natural missing visits)
  P2 missing      595 complete patients: T1 hidden (stress test), T2, T3, then T1 restored as delayed evidence (as Exp 2.8)
  P3 delayed      first 100 complete patients: T0, T2, T1 (delayed), T3 (as Exp 2.8)
  P4a correction  all complete patients with a T1 FTV: T0, T1, controlled functional correction of T1 ftv_pe_cc (+1.0 cc, a synthetic
                  marker value: tests the mechanism, not the clinical value), T2, T3
  P4b correction  complete patients whose T1 header FTV and T1 table volume (second official source) differ beyond rounding
                  (relative difference > 1e-6): the same sequence, corrected value = table volume
Run:  python run_ablation.py            (writes outputs/state_semantics_ablation/)
"""
import copy
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variants as V  # noqa: E402  (puts the Exp 2.8 package on the path)
from evaluation.continuity import state_checks  # noqa: E402
from evaluation.statistics import paired_bootstrap, patient_bootstrap  # noqa: E402
from features.measurement_features import ISPY2EvidenceSource, TP2, ispy2_cfg  # noqa: E402
from paths import EXPORTS  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

OUT = V.REPO / 'outputs/state_semantics_ablation'
RUN = OUT / 'runs'
LOG = []
STORES = {'A_full': V.VersionStore, 'B_mutable': V.MutableStore, 'D_implicit_missingness': V.VersionStore,
          'E_no_provenance': V.ProvenanceFreeStore, 'F_nondurable_correction': V.VersionStore}
CORR_FIELD, CORR_TP, CORR_SRC = 'ftv_pe_cc', 'T1', 'tumour_volume_table_cc'
CONTROLLED_OFFSET_CC, DISCREPANCY_REL = 1.0, 1e-6


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True); LOG.append(line)


# ------------------------------------------------------------------------------------------------------------------- backends
class Backend:
    def __init__(self, variant, root):
        self.v = variant
        self.store = STORES[variant](root)
        self.versioned = variant != 'B_mutable'

    def step(self, pid, tp, obs, source, cfg, event_type='scheduled', now=None):
        """One synchronization; returns (state available to later queries, written?)."""
        prev = self.store.latest(pid)
        clinical = source.clinical(pid) if prev is None else None
        if self.v == 'D_implicit_missingness':
            s, ev = V.synchronize_implicit(prev, tp, obs, clinical=clinical, cfg=cfg, event_type=event_type, now=now)
            if ev is None:
                return s, False
        else:
            s, ev = V.synchronize(prev, tp, obs, clinical=clinical, cfg=cfg, event_type=event_type, now=now)
        self.store.write(s, ev)
        return self.store.latest(pid), True

    def correct(self, pid, tp, field_name, new_value, cfg, source_label):
        prev = self.store.latest(pid)
        rv = V.revise_without_provenance if self.v == 'E_no_provenance' else V.revise
        s, rev = rv(prev, tp, field_name, new_value, reason='verified correction (controlled functional test)', actor='reviewer',
                    source=source_label, cfg=cfg, now=f'review-{tp}')
        if self.v == 'F_nondurable_correction':
            return s                                   # applied to the review-time result only; nothing is written
        self.store.write(s, dict(event_type='revision'))
        return self.store.latest(pid)


def fval(state, tp, k=CORR_FIELD):
    return (((state or {}).get('history', {}).get(tp) or {}).get('fields', {}).get(k) or {}).get('value')


def canon(x):
    return json.dumps(x, sort_keys=True, default=str)


def as_of_versions(store, pid):
    """Stored versions keyed by the protocol step at which they were written (updated_at = protocol-<tp>)."""
    try:
        return store.all_versions(pid)
    except V.NotRecoverable:
        return []


# ------------------------------------------------------------------------------------------------------------------- protocols
def p1_natural(b, source, pid, full_final, audit):
    cfg = ispy2_cfg(pid); seen = []
    for i, tp in enumerate(TP2):
        obs = source.observation(pid, tp)
        s, wrote = b.step(pid, tp, obs, source, cfg, now=f'protocol-{tp}')
        seen.append(copy.deepcopy(s))
        audit.append(dict(variant=b.v, protocol='P1_natural', patient_id=pid, step=i, timepoint=tp, event='scheduled',
                          evidence='observed' if obs else 'absent', version_written=wrote, state_version=s.get('state_version'),
                          current_timepoint=s.get('current_timepoint'), n_history=len(s['history']),
                          status_recorded=s.get('observation_status', {}).get(tp)))
    final = seen[-1]; order = list(TP2)
    obs = {tp: source.observation(pid, tp) for tp in TP2}
    missing = [tp for tp in TP2 if obs[tp] is None]
    tot = ok = 0
    for tp, o in obs.items():
        if o is None:
            continue
        for k, f in o['fields'].items():
            tot += 1; ok += canon(fval(final, tp, k)) == canon(f['value'])
    prov = V.provenance_audit(final)
    stored = as_of_versions(b.store, pid)
    # state as known at each earlier protocol step: the last stored version written at or before that step
    def as_of(k):
        cands = [s for s in stored if str(s.get('updated_at', '')).startswith('protocol-') and order.index(s['updated_at'][9:]) <= k]
        return cands[-1] if cands else None
    asof = [canon(as_of(k)) == canon(seen[k]) or (as_of(k) is None and seen[k].get('state_version') is None and not seen[k]['history'])
            for k in range(3)] if b.versioned else [False] * 3
    r = dict(variant=b.v, protocol='P1_natural', patient_id=pid,
             value_fidelity=ok / tot if tot else np.nan,
             trajectory_identical_to_full=float(canon(final['tumour_state']) == canon(full_final[pid])) if full_final else 1.0,
             provenance_complete=np.mean([x['provenance_complete'] for x in prov]) if prov else np.nan,
             source_uid_recoverable=np.mean([x['source_recoverable'] for x in prov]) if prov else np.nan,
             source_class_identifiable=np.mean([x['source_class_identifiable'] for x in prov]) if prov else np.nan,
             sync_point_known=float(final.get('current_timepoint') == TP2[-1]),
             as_of_state_reproducible=float(np.mean(asof)),
             n_missing_visits=len(missing))
    if missing:
        r['missing_recorded_explicitly'] = float(all(final.get('observation_status', {}).get(tp) == 'not_observed' for tp in missing))
        r['missing_reason_recoverable'] = float(all(bool(final.get('missingness', {}).get('visits', {}).get(tp, {}).get('reason')) for tp in missing))
        cur = final.get('current_timepoint')
        r['missing_inferable'] = float(all(tp not in final['history'] and cur in order and order.index(tp) <= order.index(cur) for tp in missing))
        r['silent_imputation'] = float(any(tp in final['history'] for tp in missing))
    return r, final, prov


def tamper_and_refusal(b, source, pid, tmp):
    """Fault injection: an unlogged overwrite of a stored historical value (outside the transition operator); and an attempted
    re-synchronization of an existing visit through the transition operator."""
    cfg = ispy2_cfg(pid)
    d = tmp / b.v / pid
    shutil.copytree(b.store.root / pid, d)
    st = STORES[b.v](tmp / b.v)
    latest = st.latest(pid)
    tp = next(t for t in TP2 if fval(latest, t) is not None) if any(fval(latest, t) is not None for t in TP2) else None
    if tp is None:
        return dict(tamper_detected=np.nan, operator_overwrite_refused=np.nan)
    f = (d / st.lineage(pid)['versions'][-1]['file']) if b.versioned else d / 'current.json'
    s = json.loads(f.read_text()); s['history'][tp]['fields'][CORR_FIELD]['value'] = fval(s, tp) * 1.5 + 1.0
    f.write_text(json.dumps(s, indent=1, sort_keys=True, default=str))
    detected = False
    if b.versioned:
        detected = not st.verify(pid)
        vs = st.all_versions(pid)
        if len(vs) > 1:
            detected |= bool(state_checks(vs[-1], vs[-2])['historical_mutation_detected'])
    before = canon(b.store.latest(pid))
    try:
        b.step(pid, tp, source.observation(pid, tp), source, cfg, event_type='late_arrival', now='overwrite-attempt')
        refused = False
    except (V.HistoricalWriteRefused, ValueError):
        refused = True
    refused &= canon(b.store.latest(pid)) == before
    shutil.rmtree(d)
    return dict(tamper_detected=float(detected), operator_overwrite_refused=float(refused))


def p2_missing(b, source, pid, audit):
    cfg = ispy2_cfg(pid); seen = []; order = list(TP2)
    for i, tp in enumerate(TP2):
        obs = source.observation(pid, tp)
        if tp == 'T1':
            obs = {'status': 'hidden_stress_test', 'reason': 'artificially hidden for Experiment 2.8 stress test'}
        s, wrote = b.step(pid, tp, obs, source, cfg, now=f'stress-scheduled-{tp}')
        seen.append(copy.deepcopy(s))
        audit.append(dict(variant=b.v, protocol='P2_missing_visit', patient_id=pid, step=i, timepoint=tp, event='scheduled',
                          evidence='hidden' if tp == 'T1' else 'observed', version_written=wrote, state_version=s.get('state_version'),
                          current_timepoint=s.get('current_timepoint'), n_history=len(s['history']),
                          status_recorded=s.get('observation_status', {}).get(tp)))
    restored = source.observation(pid, 'T1')
    final, wrote = b.step(pid, 'T1', restored, source, cfg, event_type='late_arrival', now='stress-late-arrival')
    audit.append(dict(variant=b.v, protocol='P2_missing_visit', patient_id=pid, step=4, timepoint='T1', event='late_arrival',
                      evidence='observed', version_written=wrote, state_version=final.get('state_version'),
                      current_timepoint=final.get('current_timepoint'), n_history=len(final['history']),
                      status_recorded=final.get('observation_status', {}).get('T1')))
    before = seen[-1]
    a, z = V.observation_paths(before), V.observation_paths(final)
    chk = state_checks(final, before)
    t1 = final['history'].get('T1') or {}
    later = [int(str((final['history'].get(t) or {}).get('synchronized_in', 'v-1'))[1:] or -1) for t in ('T2', 'T3')]
    t1v = str(t1.get('synchronized_in') or 'v-1')[1:]
    return dict(variant=b.v, protocol='P2_missing_visit', patient_id=pid,
                missing_known_at_occurrence=float(seen[1].get('observation_status', {}).get('T1') == 'hidden_stress_test'),
                missing_recorded_while_hidden=float(all(s.get('observation_status', {}).get('T1') == 'hidden_stress_test' for s in seen[1:])),
                missing_inferable_later=float(all('T1' not in s['history'] and order.index(s['current_timepoint']) > 1 for s in seen[2:])),
                missing_reason_recoverable=float(all((s.get('missingness', {}).get('visits', {}).get('T1') or {}).get('status') == 'hidden_stress_test' for s in seen[1:])),
                silent_imputation=float(any('T1' in s['history'] for s in seen[1:])),
                late_value_incorporated=float(fval(final, 'T1') is not None and canon(fval(final, 'T1')) == canon(restored['fields'][CORR_FIELD]['value'])),
                late_marked_explicitly=float(final.get('observation_status', {}).get('T1') == 'late_arrival'),
                late_identifiable_from_record=float(t1v.lstrip('-').isdigit() and int(t1v) > max(later)),
                unrelated_history_preserved=float(all(z.get(k) == v for k, v in a.items() if not k.startswith('history.T1.'))
                                                  and not chk['historical_mutation_detected']),
                chronology_consistent=float(final['current_timepoint'] == 'T3' and not chk['future_visit_visible']),
                lineage_integrity=float(bool(b.store.verify(pid))) if b.versioned else np.nan), final


def p3_delayed(b, source, pid, audit):
    cfg = ispy2_cfg(pid); seen = []; order = list(TP2)
    seq = [('T0', 'scheduled'), ('T2', 'scheduled'), ('T1', 'late_arrival'), ('T3', 'scheduled')]
    for i, (tp, et) in enumerate(seq):
        s, wrote = b.step(pid, tp, source.observation(pid, tp), source, cfg, event_type=et, now=f'delayed-{tp}')
        seen.append(copy.deepcopy(s))
        audit.append(dict(variant=b.v, protocol='P3_delayed', patient_id=pid, step=i, timepoint=tp, event=et, evidence='observed',
                          version_written=wrote, state_version=s.get('state_version'), current_timepoint=s.get('current_timepoint'),
                          n_history=len(s['history']), status_recorded=s.get('observation_status', {}).get(tp)))
    def preserved(states):
        ok = True
        for x, y in zip(states, states[1:]):
            o, n = V.observation_paths(x), V.observation_paths(y)
            ok &= all(k in n and n[k] == v for k, v in o.items())
        return ok
    cur = [order.index(s['current_timepoint']) for s in seen]
    stored = as_of_versions(b.store, pid)
    lin = b.store.lineage(pid)['versions']
    return dict(variant=b.v, protocol='P3_delayed', patient_id=pid,
                all_visits_present=float(set(seen[-1]['history']) == set(TP2)),
                late_marked_explicitly=float(seen[-1].get('observation_status', {}).get('T1') == 'late_arrival'),
                delayed_event_logged=float([v['event_type'] for v in lin] == [e for _, e in seq]) if b.versioned else np.nan,
                prior_observations_preserved=float(preserved(seen)),
                preservation_verifiable_from_storage=float(len(stored) == 4 and preserved(stored)),
                chronology_consistent=float(all(x <= y for x, y in zip(cur, cur[1:])) and
                                            not any(state_checks(s)['future_visit_visible'] for s in seen)),
                lineage_integrity=float(bool(b.store.verify(pid))) if b.versioned else np.nan)


def p4_correction(b, source, pid, audit, mode):
    cfg = ispy2_cfg(pid)
    s0, _ = b.step(pid, 'T0', source.observation(pid, 'T0'), source, cfg, now='protocol-T0')
    s1, _ = b.step(pid, 'T1', source.observation(pid, 'T1'), source, cfg, now='protocol-T1')
    pre = copy.deepcopy(s1); old = fval(s1, CORR_TP)
    new = old + CONTROLLED_OFFSET_CC if mode == 'P4a_correction_controlled' else fval(s1, CORR_TP, CORR_SRC)
    review = b.correct(pid, CORR_TP, CORR_FIELD, new, cfg, 'controlled functional test (+1.0 cc marker)' if mode == 'P4a_correction_controlled'
                       else f'official_trial_measurement_table:{CORR_SRC}')
    s2, _ = b.step(pid, 'T2', source.observation(pid, 'T2'), source, cfg, now='protocol-T2')
    s3, _ = b.step(pid, 'T3', source.observation(pid, 'T3'), source, cfg, now='protocol-T3')
    rows = []
    for q, s in (('review', review), ('T2', s2), ('T3', s3)):
        rows.append(dict(variant=b.v, protocol=mode, patient_id=pid, query=q, old_value=old, corrected_value=new, value_in_state=fval(s, CORR_TP),
                         reflects_correction=bool(fval(s, CORR_TP) == new)))
        audit.append(dict(variant=b.v, protocol=mode, patient_id=pid, step=q, timepoint=CORR_TP, event='query',
                          evidence='observed', version_written=None, state_version=s.get('state_version'),
                          current_timepoint=s.get('current_timepoint'), n_history=len(s['history']), status_recorded=None))
    tr = s3['tumour_state']['measures'][CORR_FIELD]['by_timepoint'].get('T2') or {}
    t2 = fval(s3, 'T2')
    stored = as_of_versions(b.store, pid)
    revs = [r for r in s3.get('revisions', []) if r['field'] == CORR_FIELD and r['timepoint'] == CORR_TP]
    fprov = ((s3['history'][CORR_TP]['fields'][CORR_FIELD]) or {}).get('provenance') or {}
    res = dict(variant=b.v, protocol=mode, patient_id=pid,
               correction_at_review=float(rows[0]['reflects_correction']),
               correction_retained_T2=float(rows[1]['reflects_correction']),
               correction_retained_T3=float(rows[2]['reflects_correction']),
               trajectory_consistent_with_correction=float(t2 is not None and tr.get('delta_from_previous') is not None
                                                           and abs(tr['delta_from_previous'] - (t2 - new)) < 1e-9),
               contradiction_review_vs_T3=float(fval(review, CORR_TP) != fval(s3, CORR_TP)),
               pre_correction_state_recoverable=float(any(canon(x) == canon(pre) for x in stored)),
               old_value_recoverable=float(any(r.get('old_value') == old for r in revs)),
               correction_source_traceable=float(any(r.get('source') for r in revs) and bool(fprov.get('revision_source'))))
    return res, rows


# ------------------------------------------------------------------------------------------------------------------- statistics
ENDPOINT_TYPE = {   # by_construction: fixed by the presence/absence of the mechanism in a deterministic system; data_dependent: depends on the cohort
    'value_fidelity': 'data_dependent', 'trajectory_identical_to_full': 'data_dependent', 'provenance_complete': 'by_construction',
    'source_uid_recoverable': 'by_construction', 'source_class_identifiable': 'by_construction', 'sync_point_known': 'data_dependent',
    'as_of_state_reproducible': 'by_construction', 'missing_recorded_explicitly': 'by_construction', 'missing_reason_recoverable': 'by_construction',
    'missing_inferable': 'data_dependent', 'silent_imputation': 'data_dependent', 'tamper_detected': 'by_construction (fault injection)',
    'operator_overwrite_refused': 'by_construction (fault injection)', 'missing_known_at_occurrence': 'by_construction',
    'missing_recorded_while_hidden': 'by_construction', 'missing_inferable_later': 'data_dependent', 'late_value_incorporated': 'data_dependent',
    'late_marked_explicitly': 'by_construction', 'late_identifiable_from_record': 'data_dependent', 'unrelated_history_preserved': 'data_dependent',
    'chronology_consistent': 'data_dependent', 'lineage_integrity': 'by_construction', 'all_visits_present': 'data_dependent',
    'delayed_event_logged': 'by_construction', 'prior_observations_preserved': 'data_dependent',
    'preservation_verifiable_from_storage': 'by_construction', 'correction_at_review': 'by_construction',
    'correction_retained_T2': 'by_construction', 'correction_retained_T3': 'by_construction',
    'trajectory_consistent_with_correction': 'data_dependent', 'contradiction_review_vs_T3': 'by_construction',
    'pre_correction_state_recoverable': 'by_construction', 'old_value_recoverable': 'by_construction',
    'correction_source_traceable': 'by_construction'}


def stats(per):
    agg, pair = [], []
    endpoints = [c for c in per.columns if c in ENDPOINT_TYPE]
    for (proto), d in per.groupby('protocol'):
        for ep in endpoints:
            if d[ep].isna().all():
                continue
            full = d[d.variant == 'A_full'].set_index('patient_id')[ep]
            for v in V.VARIANTS:
                x = d[d.variant == v].set_index('patient_id')[ep]
                if x.isna().all():
                    agg.append(dict(protocol=proto, endpoint=ep, variant=v, estimate='N/A', ci95_lo='N/A', ci95_hi='N/A', n_patients=0,
                                    endpoint_type=ENDPOINT_TYPE[ep]))
                    continue
                b = patient_bootstrap(x.values)
                agg.append(dict(protocol=proto, endpoint=ep, variant=v, estimate=b['estimate'], ci95_lo=b['ci_low'], ci95_hi=b['ci_high'],
                                n_patients=b['n_patients'], endpoint_type=ENDPOINT_TYPE[ep]))
                if v == 'A_full' or full.isna().all():
                    continue
                j = pd.concat([full, x], axis=1, keys=['a', 'b']).dropna()
                pb = paired_bootstrap(j.a.values, j.b.values)
                binary = set(np.unique(j.values)) <= {0.0, 1.0}
                nb, nc = int(((j.a == 1) & (j.b == 0)).sum()), int(((j.a == 0) & (j.b == 1)).sum())
                mc = binomtest(nb, nb + nc, .5).pvalue if binary and nb + nc else (1.0 if binary else np.nan)
                pair.append(dict(protocol=proto, endpoint=ep, comparison=f'A_full - {v}', difference=pb['estimate'], ci95_lo=pb['ci_low'],
                                 ci95_hi=pb['ci_high'], n_patients=pb['n_patients'], wilcoxon_p=pb.get('wilcoxon_p'),
                                 mcnemar_exact_p=mc, discordant_full_only=nb if binary else np.nan,
                                 discordant_variant_only=nc if binary else np.nan, endpoint_type=ENDPOINT_TYPE[ep]))
    return pd.DataFrame(agg), pd.DataFrame(pair)


# ------------------------------------------------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    if RUN.exists():
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    source = ISPY2EvidenceSource()
    all_pts = sorted(source.visits.patient_id.unique())
    manifest = pd.read_csv(EXPORTS / 'ispy2_patient_manifest.csv', dtype={'patient_id': str})
    complete = sorted(manifest.loc[manifest.complete_T0_T1_T2_T3, 'patient_id'])
    delayed = complete[:100]
    def t1(p, k):
        return ((source.observation(p, CORR_TP) or {}).get('fields', {}).get(k) or {}).get('value')
    corr_a = [p for p in complete if t1(p, CORR_FIELD) is not None]
    corr_b = [p for p in corr_a if t1(p, CORR_SRC) is not None
              and abs(t1(p, CORR_SRC) - t1(p, CORR_FIELD)) > DISCREPANCY_REL * max(abs(t1(p, CORR_FIELD)), 1e-12)]
    log(f'cohorts: natural {len(all_pts)}, missing-visit {len(complete)}, delayed {len(delayed)}, correction controlled {len(corr_a)}, '
        f'correction source-discrepancy {len(corr_b)}')
    pd.DataFrame([dict(patient_id=p, P1_natural=True, P2_missing_visit=p in complete, P3_delayed=p in delayed,
                       P4a_correction_controlled=p in corr_a, P4b_correction_source_discrepancy=p in corr_b)
                  for p in all_pts]).to_csv(OUT / 'cohort_manifest.csv', index=False)
    per, audit, corr_rows, prov_rows, full_final = [], [], [], [], {}
    tmp = Path(tempfile.mkdtemp(dir=RUN))
    for v in V.VARIANTS:
        log(f'variant {v}')
        b1 = Backend(v, RUN / v / 'P1_natural')
        for p in all_pts:
            r, final, prov = p1_natural(b1, source, p, full_final if v != 'A_full' else None, audit)
            r.update(tamper_and_refusal(b1, source, p, tmp))
            per.append(r)
            if v == 'A_full':
                full_final[p] = final['tumour_state']
            for x in prov:
                prov_rows.append(dict(variant=v, patient_id=p, **x))
        b2 = Backend(v, RUN / v / 'P2_missing_visit')
        p2_final = {}
        for p in complete:
            r, final = p2_missing(b2, source, p, audit); per.append(r); p2_final[p] = final
        b3 = Backend(v, RUN / v / 'P3_delayed')
        for p in delayed:
            per.append(p3_delayed(b3, source, p, audit))
        for mode, pts in (('P4a_correction_controlled', corr_a), ('P4b_correction_source_discrepancy', corr_b)):
            b4 = Backend(v, RUN / v / mode)
            for p in pts:
                r, rows = p4_correction(b4, source, p, audit, mode); per.append(r); corr_rows += rows
        log(f'  done {v} ({time.time() - t0:.0f} s)')
    shutil.rmtree(tmp)
    per = pd.DataFrame(per); per.to_csv(OUT / 'per_patient_results.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT / 'per_event_audit_results.csv', index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / 'correction_retention_ispy2.csv', index=False)
    prov = pd.DataFrame(prov_rows)
    prov.groupby(['variant', 'timepoint', 'field'])[['provenance_complete', 'source_recoverable', 'source_class_identifiable']].mean() \
        .reset_index().to_csv(OUT / 'provenance_audit.csv', index=False)
    per[per.protocol.isin(['P2_missing_visit', 'P3_delayed'])].dropna(axis=1, how='all') \
        .to_csv(OUT / 'missingness_late_arrival_results.csv', index=False)
    agg, pair = stats(per)
    agg.to_csv(OUT / 'aggregate_metrics_ispy2.csv', index=False); pair.to_csv(OUT / 'paired_bootstrap_statistics_ispy2.csv', index=False)
    (OUT / 'run_ispy2.log').write_text('\n'.join(LOG) + f'\nruntime {time.time() - t0:.0f} s\n')
    log(f'finished in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
