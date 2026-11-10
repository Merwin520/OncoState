"""The nine sanity checks of the ablation protocol, computed from the stored outputs (used by finalize.py and by test_ablation.py)."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variants as V  # noqa: E402
from evaluation.continuity import state_checks  # noqa: E402
from paths import EXPORTS  # noqa: E402

OUT = V.REPO / 'outputs/state_semantics_ablation'
RUN = OUT / 'runs'
PROTOCOLS = ['P1_natural', 'P2_missing_visit', 'P3_delayed', 'P4a_correction_controlled', 'P4b_correction_source_discrepancy']


def _per():
    return pd.read_csv(OUT / 'per_patient_results.csv', dtype={'patient_id': str})


def _agg():
    return pd.read_csv(OUT / 'aggregate_metrics_ispy2.csv')


def _est(protocol, endpoint, variant):
    a = _agg()
    r = a[(a.protocol == protocol) & (a.endpoint == endpoint) & (a.variant == variant)]
    return None if r.empty or r.estimate.iloc[0] == 'N/A' else float(r.estimate.iloc[0])


def stored_files(variant, protocol):
    return sorted((RUN / variant / protocol).glob('*/*.json'))


def stored_states(variant, protocol):
    return [json.loads(f.read_text()) for f in stored_files(variant, protocol) if f.name != 'lineage.json']


def check_1_same_patients():
    per = _per()
    out = {}
    for proto, d in per.groupby('protocol'):
        sets = {v: tuple(sorted(x.patient_id)) for v, x in d.groupby('variant')}
        out[proto] = len(set(sets.values())) == 1 and set(sets) == set(V.VARIANTS)
    ct = pd.read_csv(OUT / 'correction_retention_ct.csv', dtype={'patient_id': str})
    out['CT_correction_60'] = len({tuple(sorted(zip(x.patient_id, x.pb, x['query']))) for _, x in ct.groupby('variant')}) == 1
    return dict(passed=all(out.values()), detail=out)


def check_2_same_raw_evidence():
    a = pd.read_csv(OUT / 'per_event_audit_results.csv', dtype={'patient_id': str}, low_memory=False)
    out = {}
    for proto, d in a.groupby('protocol'):
        sigs = {v: tuple(map(tuple, x[['patient_id', 'step', 'timepoint', 'event', 'evidence']].astype(str).values)) for v, x in d.groupby('variant')}
        out[proto] = len(set(sigs.values())) == 1
    out['value_fidelity_all_variants'] = all(_est('P1_natural', 'value_fidelity', v) == 1.0 for v in V.VARIANTS)
    return dict(passed=all(out.values()), detail=out)


def check_3_no_future_evidence():
    bad = {}
    for v in V.VARIANTS:
        for proto in PROTOCOLS:
            n = 0
            for s in stored_states(v, proto):
                c = state_checks(s)
                if c['future_visit_visible'] or c['forbidden_outcome_fields']:
                    n += 1
            bad[f'{v}/{proto}'] = n
    return dict(passed=not any(bad.values()), detail=dict(states_with_future_or_outcome_fields=sum(bad.values())))


def check_4_full_reproduces():
    rep = json.loads((OUT / 'full_reproduction/reproduction_report.json').read_text())
    ct = pd.read_csv(OUT / 'correction_retention_ct.csv', dtype={'patient_id': str})
    arch = pd.read_csv(V.REPO / 'experiments/exp1_agentic_longitudinal/outputs/exp22_external_final/edit_retention.csv', dtype={'patient_id': str})
    arch = arch[arch.system == 'Full OncoAgent'][['patient_id', 'pb', 'pf', 'query', 'reflects_correction']].reset_index(drop=True)
    mine = ct[ct.variant == 'A_full'][['patient_id', 'pb', 'pf', 'query', 'reflects_correction']].reset_index(drop=True)
    # the ablation harness's own Full run must equal the official run on every stress-test endpoint
    per = _per()
    ms = pd.read_csv(EXPORTS / 'missingness_stress.csv', dtype={'patient_id': str}).set_index('patient_id')
    a2 = per[(per.variant == 'A_full') & (per.protocol == 'P2_missing_visit')].set_index('patient_id')
    harness = all((a2[c].astype(bool) == ms.loc[a2.index, o].astype(bool)).all() for c, o in
                  [('missing_recorded_while_hidden', 'missing_recorded_correctly'), ('silent_imputation', 'silent_imputation'),
                   ('unrelated_history_preserved', 'history_preserved'), ('lineage_integrity', 'lineage_integrity')])
    d = dict(ispy2_exp28=rep['reproduced'], ct_correction_retention=arch.astype(str).equals(mine.astype(str)),
             harness_full_equals_official_stress_test=bool(harness), n_version_files_byte_identical=rep['n_version_files'])
    return dict(passed=bool(d['ispy2_exp28'] and d['ct_correction_retention'] and harness), detail=d)


def check_5_mechanism_disabled():
    d = {}
    b = stored_files('B_mutable', 'P1_natural')
    d['B_no_lineage_or_versions'] = all(f.name == 'current.json' for f in b) and \
        all(s['state_version'] is None and s['parent_version'] is None for s in stored_states('B_mutable', 'P1_natural'))
    dst = [s for p in PROTOCOLS for s in stored_states('D_implicit_missingness', p)]
    d['D_no_typed_missingness_or_late'] = all(not s['observation_status'] and not s['missingness']['visits'] for s in dst) and \
        all(r['status'] == 'observed' for s in dst for r in s['history'].values())
    est = [s for p in PROTOCOLS for s in stored_states('E_no_provenance', p)]
    d['E_no_provenance'] = all(not s['provenance'] and s['clinical_provenance'] is None and
                               all(f.get('provenance') is None for r in s['history'].values() for f in r['fields'].values()) for s in est)
    fst = stored_states('F_nondurable_correction', 'P4a_correction_controlled')
    d['F_no_stored_revision'] = all(not s['revisions'] for s in fst)
    return dict(passed=all(d.values()), detail=d)


def check_6_unrelated_unchanged():
    e = lambda p, k, v: _est(p, k, v)  # noqa: E731
    d = dict(
        B_keeps_missingness_provenance_correction=e('P2_missing_visit', 'missing_known_at_occurrence', 'B_mutable') == 1 and
        e('P1_natural', 'provenance_complete', 'B_mutable') == 1 and e('P4a_correction_controlled', 'correction_retained_T3', 'B_mutable') == 1,
        D_keeps_lineage_provenance_correction=e('P2_missing_visit', 'lineage_integrity', 'D_implicit_missingness') == 1 and
        e('P1_natural', 'provenance_complete', 'D_implicit_missingness') == 1 and
        e('P4a_correction_controlled', 'correction_retained_T3', 'D_implicit_missingness') == 1,
        E_keeps_lineage_missingness_correction=e('P2_missing_visit', 'lineage_integrity', 'E_no_provenance') == 1 and
        e('P2_missing_visit', 'missing_known_at_occurrence', 'E_no_provenance') == 1 and
        e('P4a_correction_controlled', 'correction_retained_T3', 'E_no_provenance') == 1,
        F_keeps_lineage_missingness_provenance=e('P2_missing_visit', 'lineage_integrity', 'F_nondurable_correction') == 1 and
        e('P2_missing_visit', 'missing_known_at_occurrence', 'F_nondurable_correction') == 1 and
        e('P1_natural', 'provenance_complete', 'F_nondurable_correction') == 1)
    return dict(passed=all(d.values()), detail=d)


def check_7_values_kept():
    d = {}
    for proto in PROTOCOLS:
        a = {(s['patient_id'], s['state_version']): V.values_only(s) for s in stored_states('A_full', proto)}
        e = {(s['patient_id'], s['state_version']): V.values_only(s) for s in stored_states('E_no_provenance', proto)}
        d[proto] = a == e
    b = {s['patient_id']: V.values_only(s) for s in stored_states('B_mutable', 'P1_natural')}
    af = {}
    for s in stored_states('A_full', 'P1_natural'):
        if s['current_timepoint'] == 'T3':
            af[s['patient_id']] = V.values_only(s)
    d['B_final_values_equal_full'] = b == af
    return dict(passed=all(d.values()), detail=d)


def check_8_cohort_unchanged():
    m = pd.read_csv(EXPORTS / 'ispy2_patient_manifest.csv', dtype={'patient_id': str})
    c = pd.read_csv(OUT / 'cohort_manifest.csv', dtype={'patient_id': str})
    complete = sorted(m.loc[m.complete_T0_T1_T2_T3, 'patient_id'])
    sv = pd.read_csv(EXPORTS / 'state_versions.csv', dtype={'patient_id': str})
    d = dict(natural_equals_exp28=sorted(c.patient_id) == sorted(sv.patient_id.unique()),
             missing_equals_exp28=sorted(c.loc[c.P2_missing_visit, 'patient_id']) == complete,
             delayed_equals_exp28=sorted(c.loc[c.P3_delayed, 'patient_id']) ==
             sorted(pd.read_csv(EXPORTS / 'delayed_evidence.csv', dtype={'patient_id': str}).patient_id),
             n=dict(natural=int(len(c)), missing=int(c.P2_missing_visit.sum()), delayed=int(c.P3_delayed.sum()),
                    correction_controlled=int(c.P4a_correction_controlled.sum()), correction_source=int(c.P4b_correction_source_discrepancy.sum())))
    return dict(passed=all(v for k, v in d.items() if k != 'n'), detail=d)


def check_9_no_hidden_fallback():
    d = {}
    st = V.MutableStore(RUN / 'B_mutable' / 'P1_natural')
    pid = stored_states('B_mutable', 'P1_natural')[0]['patient_id']
    try:
        st.load(pid, 'v0'); d['B_prior_version_unloadable'] = False
    except V.NotRecoverable:
        d['B_prior_version_unloadable'] = True
    d['B_no_other_files'] = all(p.name == 'current.json' for p in (RUN / 'B_mutable').rglob('*') if p.is_file())
    words = ('"not_observed"', '"hidden_stress_test"', '"late_arrival"')
    d['D_no_typed_status_strings_in_storage'] = not any(w in f.read_text() for p in PROTOCOLS
                                                        for f in stored_files('D_implicit_missingness', p) if f.name != 'lineage.json' for w in words)
    uids = {s['source_observation']['study_uid'] for s in stored_states('A_full', 'P1_natural') if (s.get('source_observation') or {}).get('study_uid')}
    etext = ''.join(f.read_text() for f in stored_files('E_no_provenance', 'P1_natural') if f.name != 'lineage.json')
    d['E_no_source_uids_in_storage'] = not any(u in etext for u in list(uids)[:400])
    fst = stored_states('F_nondurable_correction', 'P4a_correction_controlled')
    d['F_no_revision_anywhere_in_storage'] = all('revision' not in json.dumps(s['events']) for s in fst)
    return dict(passed=all(d.values()), detail=d)


CHECKS = [check_1_same_patients, check_2_same_raw_evidence, check_3_no_future_evidence, check_4_full_reproduces, check_5_mechanism_disabled,
          check_6_unrelated_unchanged, check_7_values_kept, check_8_cohort_unchanged, check_9_no_hidden_fallback]


def run_all():
    return {c.__name__: c() for c in CHECKS}
