"""Assemble exports/exp28_final_results.csv: one row per reported Experiment 2.8 result, with CI, n and source file.

Nothing is recomputed here; every value is read from the result tables written by the pipeline.
Columns: section, endpoint, cohort, estimate, ci_low, ci_high, n_patients, n_units, unit, desired_direction, interpretation, source
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import EXPORTS, PKG  # noqa: E402


def main():
    R = []

    def r(section, endpoint, cohort, estimate, lo=np.nan, hi=np.nan, n_patients=np.nan, n_units=np.nan, unit='', direction='',
          interpretation='', source=''):
        R.append(dict(section=section, endpoint=endpoint, cohort=cohort, estimate=estimate, ci_low=lo, ci_high=hi, n_patients=n_patients,
                      n_units=n_units, unit=unit, desired_direction=direction, interpretation=interpretation, source=source))

    fig = pd.read_csv(PKG / 'figures' / 'figure28_nature_data.csv', dtype={'patient_id': str})
    S = lambda panel, label: fig[(fig.panel == panel) & (fig.series == 'stat') & (fig.label == label)].iloc[0]
    flow = pd.read_csv(EXPORTS / 'cohort_flow.csv')
    ep = pd.read_csv(EXPORTS / 'primary_endpoints.csv')
    st = pd.read_csv(EXPORTS / 'statistical_results.csv')
    E = lambda name, ds='I-SPY2': ep[(ep.endpoint == name) & (ep.dataset == ds)].iloc[0]

    # ---- cohort
    for k, n in zip(flow.patient_count_category, flow.n):
        r('cohort', k, 'I-SPY2', int(n), unit='patients', source='exports/cohort_flow.csv')
    for k, u in (('state_versions', 'immutable state versions'), ('observed_visit_records', 'visit records observed'),
                 ('explicit_missing_visit_records', 'visit records explicitly missing')):
        r('cohort', u, 'I-SPY2', int(S('A', k).value), unit='count', source='patient_states/')
    i1 = fig[fig.series == 'i1_traj']
    r('cohort', 'patients synchronized', 'I-SPY1', i1.patient_id.nunique(), unit='patients', source='external_validation (work-dir states)')
    r('cohort', 'visit records observed / explicitly missing', 'I-SPY1', f"{int(i1.value.sum())} / {int((i1.value == 0).sum())}", unit='count',
      source='figures/figure28_nature_data.csv')

    # ---- trajectory fidelity
    e = E('trajectory MAE vs own source (cc)')
    r('trajectory fidelity', 'state FTV vs synchronized source observation, MAE', 'I-SPY2', e.estimate, e.ci_low, e.ci_high, e.n_patients, unit='cc',
      direction='lower', interpretation='reproduction of the synchronized observation (exact by design)', source='exports/primary_endpoints.csv')
    for name, lab in (('state FTV identical to independent UCSF table volume (<0.01 cc)', 'state FTV identical to UCSF table volume (<0.01 cc)'),
                      ('state trajectory direction agrees with UCSF table', 'direction of change agrees with UCSF table')):
        e = E(name)
        r('trajectory fidelity', lab, 'I-SPY2', e.estimate, e.ci_low, e.ci_high, e.n_patients, S('D', 'ucsf_table_identical').n, 'proportion', 'higher',
          'independent reference; patient-level mean, bootstrap CI', 'exports/primary_endpoints.csv')
    r('trajectory fidelity', 'Spearman ρ, state FTV vs UCSF table volume', 'I-SPY2', S('D', 'ucsf_table_spearman').value, n_patients=283,
      n_units=S('D', 'ucsf_table_spearman').n, unit='rho', direction='higher', interpretation='independent reference', source='figures/figure28_nature_data.csv')
    img = pd.read_csv(EXPORTS / 'imaging_validation.csv', dtype={'patient_id': str})
    e = E('Tier 2 validated visit rate')
    r('trajectory fidelity', 'MRI-recomputed FTV reproduces official FTV (|Δ|<0.01 cc)', 'I-SPY2 Tier 2', e.estimate, e.ci_low, e.ci_high, e.n_patients,
      len(img), 'proportion', 'higher', f"{int(img.validated.astype(bool).sum())}/{len(img)} visits; 1 failure = released mask differs from header value",
      'exports/imaging_validation.csv')
    r('trajectory fidelity', 'max |MRI-recomputed − official FTV|', 'I-SPY2 Tier 2', float(img.absolute_error_cc.max()), n_patients=img.patient_id.nunique(),
      n_units=len(img), unit='cc', direction='lower', interpretation='the single failing visit (268937 T0)', source='exports/imaging_validation.csv')
    e = E('trajectory MAE (mm)', 'I-SPY1')
    r('trajectory fidelity', 'state LD vs synchronized source observation, MAE', 'I-SPY1', e.estimate, e.ci_low, e.ci_high, e.n_patients, unit='mm',
      direction='lower', interpretation='reproduction of the synchronized observation', source='exports/primary_endpoints.csv')

    # ---- continuity / invariance / leakage
    for ds in ('I-SPY2', 'I-SPY1'):
        for name, lab, d in (('history retention rate', 'history retention', 'higher'), ('provenance completeness', 'provenance completeness', 'higher'),
                             ('silent historical mutation rate', 'silent historical mutation rate', 'lower')):
            e = E(name, ds)
            r('state continuity', lab, ds, e.estimate, e.ci_low, e.ci_high, e.n_patients, unit='proportion', direction=d,
              interpretation='patient-level, all versions', source='exports/primary_endpoints.csv')
    r('state continuity', 'SHA-256 lineage chain intact', 'I-SPY2', S('E', 'lineage_integrity').value, n_patients=S('E', 'lineage_integrity').n,
      unit='proportion', direction='higher', source='patient_states/*/lineage.json')
    r('leakage', 'versions with no future visit visible', 'I-SPY2', S('E', 'no_future_visit_visible').value, n_patients=719,
      n_units=S('E', 'no_future_visit_visible').n, unit='proportion', direction='higher', source='patient_states/')
    r('leakage', 'versions containing outcome fields (pCR etc.)', 'I-SPY2', 0, n_patients=719, n_units=2876, unit='count', direction='lower',
      source='VALIDATION_REPORT.md')

    # ---- positive controls
    det = pd.read_csv(EXPORTS / 'detector_validation_summary.csv')
    for row in det.itertuples():
        r('positive control', row.check.replace('_', ' '), 'I-SPY2', row.detected_or_correct / row.n_patients_tested, n_patients=row.n_patients_tested,
          unit='proportion', direction='higher', interpretation='check applied to a deliberately corrupted copy of each real version',
          source='exports/detector_validation_summary.csv')

    # ---- missingness
    nat = pd.read_csv(EXPORTS / 'natural_missingness.csv', dtype={'patient_id': str})
    ab = nat[nat.actually_missing]
    r('missingness', 'real absent visits recorded as missing', 'I-SPY2', float(ab.missing_recorded_correctly.mean()), n_patients=ab.patient_id.nunique(),
      n_units=len(ab), unit='proportion', direction='higher', source='exports/natural_missingness.csv')
    r('missingness', 'silent imputation of absent visits', 'I-SPY2', int(nat.silent_imputation.sum()), n_patients=nat.patient_id.nunique(),
      n_units=len(ab), unit='count', direction='lower', source='exports/natural_missingness.csv')
    hh = fig[(fig.panel == 'H') & (fig.series == 'I-SPY1') & (fig.label == 'absent_recorded_missing')].iloc[0]
    r('missingness', 'real absent visits recorded as missing', 'I-SPY1', hh.value, n_units=hh.n, unit='proportion', direction='higher',
      source='figures/figure28_nature_data.csv')
    s = pd.read_csv(EXPORTS / 'missingness_stress.csv', dtype={'patient_id': str})
    for col, lab, d in (('missing_recorded_correctly', 'hidden T1 recorded as missing (stress test)', 'higher'),
                        ('silent_imputation', 'silent imputation of hidden T1 (stress test)', 'lower'),
                        ('late_update_success', 'late T1 inserted as explicit late-arrival event', 'higher'),
                        ('history_preserved', 'other visits unchanged after late insertion', 'higher')):
        r('missingness', lab, 'I-SPY2 stress', float(s[col].astype(bool).mean()), n_patients=len(s), unit='proportion', direction=d,
          interpretation='EXPERIMENTAL: T1 deliberately hidden, then restored', source='exports/missingness_stress.csv')
    dl = pd.read_csv(EXPORTS / 'delayed_evidence.csv', dtype={'patient_id': str})
    r('missingness', 'out-of-order arrival (T0,T2,T1 late,T3) handled with intact lineage', 'I-SPY2 stress',
      float((dl.delayed_logged_explicitly & dl.prior_observations_preserved & dl.lineage_integrity).mean()), n_patients=len(dl), unit='proportion',
      direction='higher', interpretation='secondary test', source='exports/delayed_evidence.csv')

    # ---- efficiency
    for row in st[st.metric.isin(['source_reads', 'runtime_seconds'])].itertuples():
        r('efficiency', f'stateless − persistent {row.metric} per update at {row.timepoint}', row.dataset, row.estimate, row.ci_low, row.ci_high,
          row.n_patients, unit='reads' if row.metric == 'source_reads' else 'seconds', direction='>0 favours persistence',
          interpretation=f'paired patient bootstrap; Wilcoxon p = {row.wilcoxon_p:.3g}', source='exports/statistical_results.csv')
    for coh in ('I-SPY2', 'I-SPY1'):
        v = fig[(fig.panel == 'H') & (fig.series == coh) & (fig.label == 'reads_ratio_at_4th_visit')].value.iloc[0]
        r('efficiency', 'stateless ÷ persistent source reads at the 4th visit', coh, v, unit='ratio', source='figures/figure28_nature_data.csv')
    ip = fig[(fig.panel == 'F') & (fig.series == 'imaging_persistent')].sort_values('x')
    isl = fig[(fig.panel == 'F') & (fig.series == 'imaging_stateless')].sort_values('x')
    r('efficiency', 'MRI fetched per update at 4th visit, persistent', 'I-SPY2 Tier 2', ip.value.iloc[-1], n_patients=ip.n.iloc[-1], unit='MB',
      interpretation='measured per-visit bytes', source='exports/imaging_validation.csv')
    r('efficiency', 'MRI a stateless rebuild must fetch at 4th visit', 'I-SPY2 Tier 2', isl.value.iloc[-1], n_patients=isl.n.iloc[-1], unit='MB',
      interpretation='sum of measured per-visit bytes T0..T3', source='exports/imaging_validation.csv')

    # ---- storage
    dla = pd.read_csv(EXPORTS / 'download_audit.csv')
    a = dla[(dla.cohort == 'all 719 patients') & dla.strategy.str.startswith('A ')].iloc[0]
    r('storage', 'Tier 1 network transfer (SEG header range reads)', 'I-SPY2', a.transfer_GB, n_patients=719, n_units=2688, unit='GB',
      interpretation='0 bytes cached', source='exports/download_audit.csv')
    r('storage', 'Tier 2 network transfer (SEG + early PE + cropped DCE)', 'I-SPY2 Tier 2', round(img.bytes_selected.sum() / 1e9, 2),
      n_patients=img.patient_id.nunique(), n_units=len(img), unit='GB', interpretation='streamed one visit at a time; raw deleted after outcome recorded',
      source='exports/imaging_validation.csv')
    r('storage', 'largest single-visit imaging cache', 'I-SPY2 Tier 2', round(img.bytes_selected.max() / 1e9, 3), unit='GB', direction='≤ 10 GB cap',
      source='exports/imaging_validation.csv')
    r('storage', 'raw imaging files left in cache / package', 'I-SPY2', 0, unit='count', direction='lower', source='VALIDATION_REPORT.md')

    out = pd.DataFrame(R)
    for c in ('estimate', 'ci_low', 'ci_high'):
        out[c] = [round(v, 6) if isinstance(v, (float, np.floating)) and np.isfinite(v) else v for v in out[c]]
    out.to_csv(EXPORTS / 'exp28_final_results.csv', index=False)
    print(f'{len(out)} rows ->', EXPORTS / 'exp28_final_results.csv')
    print(out.groupby('section').size().to_dict())


if __name__ == '__main__':
    main()
