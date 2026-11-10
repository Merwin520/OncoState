#!/usr/bin/env python3
"""Assemble the deliverables of the state-semantics ablation from the run outputs (no number is typed in):
aggregate_metrics.csv, paired_bootstrap_statistics.csv, correction_retention.csv, summary_table.{csv,md}, figure_source_data.csv,
sanity_checks.json, ablation_config.yaml, environment.txt, git_commit.txt."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanity  # noqa: E402
import variants as V  # noqa: E402
from evaluation.statistics import paired_bootstrap, patient_bootstrap  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

OUT = V.REPO / 'outputs/state_semantics_ablation'
NAMES = {'A_full': 'Full OncoState', 'B_mutable': 'Mutable State', 'C_no_identity': 'No Formal Persistent Lesion Identity',
         'D_implicit_missingness': 'Implicit Missingness', 'E_no_provenance': 'No Field-Level Provenance',
         'F_nondurable_correction': 'Non-Durable Correction'}
REMOVED = {'A_full': '— (reference)', 'B_mutable': 'write-once versions, parent chain, SHA-256 lineage',
           'C_no_identity': 'canonical cross-visit lesion object', 'D_implicit_missingness': 'typed missingness and late-arrival transition',
           'E_no_provenance': 'field-level provenance and source identifiers', 'F_nondurable_correction': 'commit of the correction to the state'}
C_REASON = ('Not separable in the existing code/data. The locked CT cohort is two-timepoint (baseline, follow-up); its official state stores '
            'visit-local lesion IDs plus pairwise correspondence rows, so there is no canonical cross-visit lesion object distinct from the '
            'correspondence to remove (identity = correspondence with two visits). I-SPY2/I-SPY1 states are tumour-level (no lesion objects). '
            'Implementing C would require new >=3-timepoint lesion-level data or a redesign of the CT state; per protocol it was skipped.')


def cluster_ratio_bootstrap(num, den, seed=20261001, n=2000):
    """Patient-cluster bootstrap of a pooled proportion sum(num)/sum(den) (same seed and resamples as the Exp 2.8 procedure)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    g = np.random.default_rng(seed); k = len(num)
    bs = []
    for _ in range(n):
        i = g.integers(0, k, k); bs.append(num[i].sum() / den[i].sum())
    return float(num.sum() / den.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), bs


def ct_stats():
    """CT arm of F. Correction retention = later queries (Q2, Q3) using the corrected value / later queries (query-level pooled rate,
    patient-cluster bootstrap); contradiction = events whose review answer reflects the correction but a later answer does not."""
    ct = pd.read_csv(OUT / 'correction_retention_ct.csv', dtype={'patient_id': str})
    later = ct[ct['query'].isin(['Q2', 'Q3'])]
    q1 = ct[ct['query'] == 'Q1'].set_index(['variant', 'patient_id', 'pb', 'pf']).reflects_correction
    lat = later.groupby(['variant', 'patient_id', 'pb', 'pf']).reflects_correction.min()
    contra = (q1 & ~lat).rename('contradiction').reset_index()
    tabs = {'correction_retention_Q2_Q3': later.assign(x=later.reflects_correction.astype(float)),
            'contradiction_review_vs_later': contra.assign(x=contra.contradiction.astype(float))}
    agg, pair = [], []
    for ep, t in tabs.items():
        sums = t.groupby(['variant', 'patient_id']).x.agg(['sum', 'count']).unstack(0)
        res = {}
        for v in ['A_full', 'F_nondurable_correction']:
            e, lo, hi, bs = cluster_ratio_bootstrap(sums['sum'][v].values, sums['count'][v].values)
            res[v] = (e, np.asarray(bs))
            agg.append(dict(protocol='CT_correction_60', endpoint=ep, variant=v, estimate=e, ci95_lo=lo, ci95_hi=hi,
                            n_patients=len(sums), endpoint_type='by_construction'))
        for v in ['B_mutable', 'D_implicit_missingness', 'E_no_provenance']:
            agg.append(dict(protocol='CT_correction_60', endpoint=ep, variant=v, estimate='N/A', ci95_lo='N/A', ci95_hi='N/A', n_patients=0,
                            endpoint_type='not applicable (mechanism absent from the CT state)'))
        d = res['A_full'][1] - res['F_nondurable_correction'][1]      # same resampled patients (same seed) -> paired
        pair.append(dict(protocol='CT_correction_60', endpoint=ep, comparison='A_full - F_nondurable_correction',
                         difference=res['A_full'][0] - res['F_nondurable_correction'][0], ci95_lo=float(np.quantile(d, .025)),
                         ci95_hi=float(np.quantile(d, .975)), n_patients=len(sums), wilcoxon_p=np.nan, mcnemar_exact_p=np.nan,
                         discordant_full_only=np.nan, discordant_variant_only=np.nan, endpoint_type='by_construction'))
    # patient-level exact McNemar (exp22b_stats.mcnemar_rows): patient correct = all later queries reflect it
    pp = later.groupby(['variant', 'patient_id']).reflects_correction.all().unstack(0)
    nb, nc = int((pp.A_full & ~pp.F_nondurable_correction).sum()), int((~pp.A_full & pp.F_nondurable_correction).sum())
    pair[0].update(mcnemar_exact_p=binomtest(nb, nb + nc, .5).pvalue if nb + nc else 1.0, discordant_full_only=nb, discordant_variant_only=nc)
    return pd.DataFrame(agg), pd.DataFrame(pair)


def correction_table():
    isp = pd.read_csv(OUT / 'correction_retention_ispy2.csv', dtype={'patient_id': str})
    isp['query'] = isp['query'].map({'review': 'review', 'T2': 'later_T2', 'T3': 'later_T3'})
    isp['silent_reversion_to_pre_correction'] = (isp.value_in_state == isp.old_value) & isp['query'].str.startswith('later')
    isp = isp.rename(columns={'reflects_correction': 'uses_corrected_value'})
    ct = pd.read_csv(OUT / 'correction_retention_ct.csv', dtype={'patient_id': str})
    ct['query'] = ct['query'].map({'Q1': 'review', 'Q2': 'later_Q2', 'Q3': 'later_Q3'})
    ct['silent_reversion_to_pre_correction'] = ~ct.reflects_correction & ct['query'].str.startswith('later')
    ct = ct.rename(columns={'reflects_correction': 'uses_corrected_value'})
    rows = pd.concat([isp.assign(dataset='I-SPY2'), ct.assign(dataset='CT (Exp 2.2)')], ignore_index=True)
    rows.to_csv(OUT / 'correction_retention.csv', index=False)
    later = rows[rows['query'].str.startswith('later')]
    s = later.groupby(['dataset', 'protocol', 'variant']).agg(subsequent_queries=('uses_corrected_value', 'size'),
                                                              using_corrected_value=('uses_corrected_value', 'sum'),
                                                              silent_reversions=('silent_reversion_to_pre_correction', 'sum')).reset_index()
    s['correction_retention_rate'] = s.using_corrected_value / s.subsequent_queries
    s.to_csv(OUT / 'correction_retention_summary.csv', index=False)
    return s


def fmt(agg, proto, ep, v, n=True):
    r = agg[(agg.protocol == proto) & (agg.endpoint == ep) & (agg.variant == v)]
    if r.empty or str(r.estimate.iloc[0]) == 'N/A':
        return 'N/A'
    e, lo, hi, k = (float(r.estimate.iloc[0]), float(r.ci95_lo.iloc[0]), float(r.ci95_hi.iloc[0]), int(r.n_patients.iloc[0]))
    return f'{e:.3f} [{lo:.3f}, {hi:.3f}]' + (f' (n={k})' if n else '')


def summary(agg, cs):
    rows = []
    for v in ['A_full', 'B_mutable', 'C_no_identity', 'D_implicit_missingness', 'E_no_provenance', 'F_nondurable_correction']:
        if v == 'C_no_identity':
            rows.append({'Variant': NAMES[v], 'Primary mechanism removed': REMOVED[v], **{k: 'not run (not separable; see config)' for k in
                         ['Matching / longitudinal metric', 'Trajectory consistency', 'Correction retention', 'History mutation detected',
                          'Missing/late-arrival correctness', 'Provenance traceability']}})
            continue
        def cr(ds, proto):
            r = cs[(cs.dataset == ds) & (cs.protocol == proto) & (cs.variant == v)]
            return 'N/A' if r.empty else f'{r.correction_retention_rate.iloc[0]:.3f} ({int(r.using_corrected_value.iloc[0])}/{int(r.subsequent_queries.iloc[0])})'
        rows.append({'Variant': NAMES[v], 'Primary mechanism removed': REMOVED[v],
                     'Matching / longitudinal metric': 'value fidelity ' + fmt(agg, 'P1_natural', 'value_fidelity', v),
                     'Trajectory consistency': 'after correction ' + fmt(agg, 'P4a_correction_controlled', 'trajectory_consistent_with_correction', v, False)
                     + '; chronology after late arrival ' + fmt(agg, 'P2_missing_visit', 'chronology_consistent', v, False),
                     'Correction retention': f"I-SPY2 {cr('I-SPY2', 'P4a_correction_controlled')}; CT {cr('CT (Exp 2.2)', 'CT_correction_60')}",
                     'History mutation detected': fmt(agg, 'P1_natural', 'tamper_detected', v)
                     + '; prior state recoverable ' + fmt(agg, 'P1_natural', 'as_of_state_reproducible', v, False),
                     'Missing/late-arrival correctness': 'missing recorded at occurrence ' + fmt(agg, 'P2_missing_visit', 'missing_known_at_occurrence', v, False)
                     + '; natural missing visit inferable ' + fmt(agg, 'P1_natural', 'missing_inferable', v, False)
                     + '; late value incorporated ' + fmt(agg, 'P2_missing_visit', 'late_value_incorporated', v, False)
                     + '; late arrival typed ' + fmt(agg, 'P2_missing_visit', 'late_marked_explicitly', v, False),
                     'Provenance traceability': fmt(agg, 'P1_natural', 'source_uid_recoverable', v)
                     + '; correction source ' + fmt(agg, 'P4a_correction_controlled', 'correction_source_traceable', v, False)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'summary_table.csv', index=False)
    md = ['| ' + ' | '.join(t.columns) + ' |', '|' + '---|' * len(t.columns)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in t.values]
    (OUT / 'summary_table.md').write_text('\n'.join(md) + '\n\nEstimates are patient-level means with 95% patient-bootstrap CIs (2000 '
                                          'resamples, seed 20261001). N/A = endpoint not applicable (mechanism absent from that state/cohort).\n')
    return t


# figure source data: one row per (variant, endpoint) cell of the matrix and per paired difference
FIG_COLUMNS = [  # (group, protocol, endpoint, label) of matrix panel a
    ('Values', 'P1_natural', 'value_fidelity', 'Stored values correct'),
    ('Versioned history', 'P1_natural', 'tamper_detected', 'Silent overwrite detected'),
    ('Versioned history', 'P1_natural', 'as_of_state_reproducible', 'Earlier state recoverable'),
    ('Missingness and late arrival', 'P2_missing_visit', 'missing_known_at_occurrence', 'Missing visit recorded'),
    ('Missingness and late arrival', 'P1_natural', 'missing_inferable', 'Missing visit inferable'),
    ('Missingness and late arrival', 'P1_natural', 'sync_point_known', 'Synchronization point known'),
    ('Missingness and late arrival', 'P2_missing_visit', 'late_value_incorporated', 'Late value incorporated'),
    ('Missingness and late arrival', 'P2_missing_visit', 'late_marked_explicitly', 'Late arrival typed'),
    ('Missingness and late arrival', 'P2_missing_visit', 'chronology_consistent', 'Chronology consistent'),
    ('Provenance', 'P1_natural', 'source_uid_recoverable', 'Value traceable to source'),
    ('Provenance', 'P4a_correction_controlled', 'correction_source_traceable', 'Correction source traceable'),
    ('Correction', 'P4a_correction_controlled', 'correction_retained_T3', 'Kept at later visit (I-SPY2)'),
    ('Correction', 'CT_correction_60', 'correction_retention_Q2_Q3', 'Kept in later queries (CT)'),
    ('Correction', 'P4a_correction_controlled', 'trajectory_consistent_with_correction', 'Trajectory uses correction'),
]


def clopper(k, n):
    from scipy.stats import beta
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))


def figure_source(agg, pair):
    rows = []
    for g, proto, ep, lab in FIG_COLUMNS:
        for v in V.VARIANTS:
            r = agg[(agg.protocol == proto) & (agg.endpoint == ep) & (agg.variant == v)].iloc[0]
            p = pair[(pair.protocol == proto) & (pair.endpoint == ep) & (pair.comparison == f'A_full - {v}')]
            na = str(r.estimate) == 'N/A'
            rows.append(dict(panel='a', group=g, protocol=proto, endpoint=ep, label=lab, variant=v, variant_name=NAMES[v],
                             estimate=np.nan if na else float(r.estimate), ci95_lo=np.nan if na else float(r.ci95_lo),
                             ci95_hi=np.nan if na else float(r.ci95_hi), n_patients=int(r.n_patients), applicable=not na,
                             endpoint_type=r.endpoint_type,
                             diff_full_minus_variant=np.nan if p.empty else float(p.difference.iloc[0]),
                             diff_ci95_lo=np.nan if p.empty else float(p.ci95_lo.iloc[0]),
                             diff_ci95_hi=np.nan if p.empty else float(p.ci95_hi.iloc[0])))
        rows.append(dict(panel='a', group=g, protocol=proto, endpoint=ep, label=lab, variant='C_no_identity', variant_name=NAMES['C_no_identity'],
                         estimate=np.nan, applicable=False, endpoint_type='not run (not separable)'))
    # panel c: fraction of answers using the verified correction, per query (exact Clopper-Pearson 95% CI)
    cr = pd.read_csv(OUT / 'correction_retention.csv', dtype={'patient_id': str})
    for (ds, proto, v, q), d in cr.groupby(['dataset', 'protocol', 'variant', 'query']):
        if proto == 'P4b_correction_source_discrepancy':
            continue
        k, n = int(d.uses_corrected_value.sum()), int(len(d)); lo, hi = clopper(k, n)
        rows.append(dict(panel='c', group=ds, protocol=proto, endpoint='uses_corrected_value', label=q, variant=v, variant_name=NAMES[v],
                         estimate=k / n, ci95_lo=lo, ci95_hi=hi, n_patients=d.patient_id.nunique(), n_queries=n, k_queries=k, applicable=True,
                         endpoint_type='by_construction'))
    # panel d: sequence-level state error rate (sequence_endpoint.py), per scenario and per question
    sq = OUT / 'sequence_aggregate.csv'
    if sq.exists():
        sa = pd.read_csv(sq)
        for r in sa.itertuples():
            rows.append(dict(panel='d', group=r.scenario, protocol='sequence_stream', endpoint=r.endpoint, label=r.endpoint, variant=r.variant,
                             variant_name=NAMES[r.variant], estimate=r.estimate, ci95_lo=r.ci95_lo, ci95_hi=r.ci95_hi, n_patients=r.n_patients,
                             applicable=True, endpoint_type='data_dependent (simulated event rates, real visit data)'))
    li = OUT / 'lesion_identity/aggregate.csv'
    if li.exists():
        la = pd.read_csv(li); lp = pd.read_csv(OUT / 'lesion_identity/paired_bootstrap.csv').set_index('miss_rate')
        for r in la.itertuples():
            v = 'A_full' if r.condition == 'A_canonical_identity' else 'C_no_identity'
            rows.append(dict(panel='e', group='HCUCH multi-visit CT', protocol='lesion_identity', endpoint='identity_preserved', label=str(r.miss_rate),
                             variant=v, variant_name=NAMES[v], estimate=r.estimate, ci95_lo=r.ci95_lo, ci95_hi=r.ci95_hi, n_patients=r.n_patients,
                             applicable=True, endpoint_type='data_dependent (simulated detector misses on real multi-visit CT)',
                             diff_full_minus_variant=lp.at[r.miss_rate, 'difference'], diff_ci95_lo=lp.at[r.miss_rate, 'ci95_lo'],
                             diff_ci95_hi=lp.at[r.miss_rate, 'ci95_hi']))
    pd.DataFrame(rows).to_csv(OUT / 'figure_source_data.csv', index=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance_files():
    off = sorted((V.REPO / 'experiments/exp28_repeated_synchronization/state').glob('*.py')) + \
        sorted((V.REPO / 'experiments/exp28_repeated_synchronization/stress_tests').glob('*.py')) + \
        [V.REPO / 'experiments/exp28_repeated_synchronization/evaluation/continuity.py',
         V.REPO / 'experiments/exp28_repeated_synchronization/evaluation/statistics.py',
         V.REPO / 'experiments/exp28_repeated_synchronization/features/measurement_features.py',
         V.REPO / 'experiments/exp1_agentic_longitudinal/exp22b_retention.py', V.REPO / 'experiments/exp1_agentic_longitudinal/exp22b_common.py']
    mine = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh'))
    return off, mine


def main():
    ai = pd.read_csv(OUT / 'aggregate_metrics_ispy2.csv'); pi = pd.read_csv(OUT / 'paired_bootstrap_statistics_ispy2.csv')
    ac, pc = ct_stats()
    agg = pd.concat([ai, ac], ignore_index=True); pair = pd.concat([pi, pc], ignore_index=True)
    agg.to_csv(OUT / 'aggregate_metrics.csv', index=False); pair.to_csv(OUT / 'paired_bootstrap_statistics.csv', index=False)
    cs = correction_table()
    t = summary(agg, cs)
    figure_source(agg, pair)
    sc = sanity.run_all()
    sc['all_passed'] = all(v['passed'] for v in sc.values())
    (OUT / 'sanity_checks.json').write_text(json.dumps(sc, indent=1, default=str))
    off, mine = provenance_files()
    cfg = dict(
        study='OncoState state-semantics ablation',
        full_implementation=dict(code='experiments/exp28_repeated_synchronization (state/synchronization.py synchronize/revise, '
                                      'state/versioning.py VersionStore, state/provenance.py) and experiments/exp1_agentic_longitudinal/exp22b_retention.py',
                                 modified=False, sha256={str(p.relative_to(V.REPO)): sha(p) for p in off}),
        variants={v: dict(name=NAMES[v], removed=REMOVED[v]) for v in NAMES},
        skipped=dict(C_no_identity=C_REASON),
        not_applicable_on_CT='B, D and E: the Exp 2.2 CT state has no version store, no typed missingness and no field-level provenance; only F is defined on CT.',
        protocols=dict(P1_natural='all I-SPY2 patients, scheduled T0..T3 with natural missing visits',
                       P2_missing_visit='Exp 2.8 missing-visit stress test (T1 hidden, restored as delayed evidence), complete patients',
                       P3_delayed='Exp 2.8 delayed-evidence test T0,T2,T1(late),T3, first 100 complete patients',
                       P4a_correction_controlled='T0,T1, correction of T1 ftv_pe_cc by +1.0 cc (synthetic marker value; functional test of the mechanism), T2,T3',
                       P4b_correction_source_discrepancy='same sequence; corrected value = T1 table tumour volume where it differs from the header FTV beyond rounding (relative > 1e-6)',
                       CT_correction_60='the 60 real matching-error correction events of exp22b_retention (seed 20260921); Q1 review, Q2-Q3 later'),
        fault_injection=dict(silent_overwrite='after P1, the stored T-first-observed ftv_pe_cc is overwritten (x1.5 + 1) directly in storage, outside the transition operator; detection = SHA-256 lineage verify() or state_checks against the parent version',
                             operator_overwrite='re-synchronization of an existing visit through the transition operator (event_type late_arrival)'),
        statistics=dict(procedure='experiments/exp28_repeated_synchronization/evaluation/statistics.py patient_bootstrap / paired_bootstrap',
                        resamples=2000, seed=20261001, difference='Full - variant', binary_test='exact McNemar (binomial on discordant pairs)',
                        note='Endpoints marked by_construction are determined by the presence/absence of the mechanism in this deterministic system; their CIs are degenerate and they are design verifications, not estimates of behaviour.'),
        D_design='conventional upsert: absent evidence leaves no record; delayed evidence stored as observed; current_timepoint = latest observed protocol visit (never moves backwards); write guard on existing visits kept',
        E_design='values, units, QC flags, mask_reference value kept; field provenance, provenance log, clinical provenance, record/source-observation study and series UIDs, revision source/actor removed',
        F_design='verified correction applied to the review-time result only; not written to the store',
        outputs_dir='outputs/state_semantics_ablation', official_outputs_overwritten=False,
        code_sha256={str(p.relative_to(V.REPO)): sha(p) for p in mine})
    (OUT / 'ablation_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, width=160))
    env = [f'python {sys.version}', f'platform {platform.platform()}', f'executable {sys.executable}', '']
    env += subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout.splitlines()
    (OUT / 'environment.txt').write_text('\n'.join(env) + '\n')
    g = subprocess.run(['git', '-C', str(V.REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    (OUT / 'git_commit.txt').write_text((g.stdout.strip() if g.returncode == 0 else
                                         'not a git repository: no commit hash available; code identity is recorded as SHA-256 below') + '\n' +
                                        '\n'.join(f'{sha(p)}  {p.relative_to(V.REPO)}' for p in off + mine) + '\n')
    print(t.to_string(index=False)); print('sanity all passed:', sc['all_passed'])


if __name__ == '__main__':
    main()
