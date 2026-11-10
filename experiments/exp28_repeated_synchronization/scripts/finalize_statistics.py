"""Create endpoint and patient-level bootstrap tables from completed runs."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.efficiency import paired_by_timepoint  # noqa: E402
from evaluation.missingness import patient_accuracy  # noqa: E402
from evaluation.statistics import paired_bootstrap, patient_bootstrap  # noqa: E402
from evaluation.trajectory import direction_metrics, patient_metrics  # noqa: E402
from paths import EXPORTS  # noqa: E402


def add(rows, endpoint, values, dataset='I-SPY2', desired='higher', notes=''):
    result = patient_bootstrap(values)
    rows.append(dict(dataset=dataset, endpoint=endpoint, desired_direction=desired, notes=notes, **result))


def main():
    fidelity = pd.read_csv(EXPORTS / 'trajectory_fidelity.csv', dtype={'patient_id': str})
    state = pd.read_csv(EXPORTS / 'state_versions.csv', dtype={'patient_id': str})
    natural = pd.read_csv(EXPORTS / 'natural_missingness.csv', dtype={'patient_id': str})
    stress = pd.read_csv(EXPORTS / 'missingness_stress.csv', dtype={'patient_id': str})
    efficiency = pd.read_csv(EXPORTS / 'sync_efficiency.csv', dtype={'patient_id': str})
    rows = []
    ftv = fidelity[fidelity.measure == 'ftv_pe_cc']
    add(rows, 'trajectory MAE vs own source (cc)', patient_metrics(ftv).mae, desired='lower', notes='reproduction of the synchronized observation')
    add(rows, 'trajectory direction agreement vs own source', direction_metrics(ftv).direction_accuracy, notes='reproduction of the synchronized observation')
    add(rows, 'history retention rate', state.groupby('patient_id').history_retention_rate.min())
    add(rows, 'provenance completeness', state.groupby('patient_id').provenance_complete.mean())
    add(rows, 'silent historical mutation rate', state.groupby('patient_id').historical_mutation_detected.mean(), desired='lower')
    missing = patient_accuracy(natural)
    add(rows, 'natural missingness detection accuracy', missing.missingness_detection_accuracy)
    add(rows, 'natural silent imputation rate', missing.silent_imputation_rate, desired='lower')
    add(rows, 'stress late-update success', stress.groupby('patient_id').late_update_success.mean())
    add(rows, 'stress history preserved', stress.groupby('patient_id').history_preserved.mean())
    add(rows, 'stress silent imputation rate', stress.groupby('patient_id').silent_imputation.mean(), desired='lower')
    statistics = []
    for timepoint in ['T0', 'T1', 'T2', 'T3']:
        for metric in ['source_reads', 'series_reads', 'tool_calls', 'state_ops', 'runtime_seconds']:
            paired = paired_by_timepoint(efficiency[efficiency.timepoint == timepoint], metric)
            result = paired_bootstrap(paired.stateless, paired.persistent)
            statistics.append(dict(dataset='I-SPY2', timepoint=timepoint, metric=metric,
                                   contrast='stateless - persistent', **result))
    # trajectory fidelity against an INDEPENDENT reference (UCSF multi-feature table), from the final state versions
    import numpy as np
    from paths import STATES
    from state.versioning import VersionStore
    store = VersionStore(STATES)
    identical, direction = [], []
    for patient in sorted(p.name for p in STATES.iterdir() if p.is_dir()):
        final = store.latest(patient)
        pairs = [(final['history'][t]['fields']['ftv_pe_cc']['value'], final['history'][t]['fields']['tumour_volume_table_cc']['value'])
                 for t in ['T0', 'T1', 'T2', 'T3'] if t in final['history'] and 'tumour_volume_table_cc' in final['history'][t]['fields']
                 and final['history'][t]['fields']['tumour_volume_table_cc']['value'] is not None]
        if pairs:
            s, r = np.array(pairs, float).T
            identical.append(float(np.mean(np.abs(s - r) < .01)))
            if len(s) > 1:
                direction.append(float(np.mean(np.sign(np.diff(s)) == np.sign(np.diff(r)))))
    add(rows, 'state FTV identical to independent UCSF table volume (<0.01 cc)', identical, notes='independent reference')
    add(rows, 'state trajectory direction agrees with UCSF table', direction, notes='independent reference')
    i1_eff_path = EXPORTS / 'ispy1_sync_efficiency.csv'
    if i1_eff_path.exists():
        i1_eff = pd.read_csv(i1_eff_path, dtype={'patient_id': str})
        for timepoint in ['V0', 'V1', 'V2', 'V3']:
            for metric in ['source_reads', 'tool_calls', 'state_ops', 'runtime_seconds']:
                paired = paired_by_timepoint(i1_eff[i1_eff.timepoint == timepoint], metric)
                statistics.append(dict(dataset='I-SPY1', timepoint=timepoint, metric=metric, contrast='stateless - persistent',
                                       **paired_bootstrap(paired.stateless, paired.persistent)))
    detector_path = EXPORTS / 'detector_validation.csv'
    if detector_path.exists():
        detector = pd.read_csv(detector_path, dtype={'patient_id': str})
        for column in detector.columns.drop('patient_id'):
            add(rows, f'positive control: {column}', detector[column].dropna().astype(float), notes='corrupted copies of real versions')
    natural_absent = natural[natural.actually_missing]
    add(rows, 'natural absent visits recorded as missing', natural_absent.groupby('patient_id').missing_recorded_correctly.mean(),
        notes=f'{len(natural_absent)} absent visits')
    external_path = EXPORTS / 'ispy1_external_replication.csv'
    if external_path.exists():
        external = pd.read_csv(external_path, dtype={'patient_id': str})
        version_rows = external[external.state_version != 'final_check']
        check_rows = external[external.state_version == 'final_check']
        add(rows, 'trajectory MAE (mm)', check_rows.groupby('patient_id').absolute_error.mean(), 'I-SPY1', 'lower')
        add(rows, 'history retention rate', version_rows.groupby('patient_id').history_retention_rate.min(), 'I-SPY1')
        add(rows, 'provenance completeness', version_rows.groupby('patient_id').provenance_complete.mean(), 'I-SPY1')
        add(rows, 'silent historical mutation rate', version_rows.groupby('patient_id').historical_mutation_detected.mean(), 'I-SPY1', 'lower')
    imaging_path = EXPORTS / 'imaging_validation.csv'
    if imaging_path.exists():
        imaging = pd.read_csv(imaging_path, dtype={'patient_id': str})
        add(rows, 'Tier 2 local FTV absolute error (cc)', imaging.groupby('patient_id').absolute_error_cc.mean(),
            desired='lower', notes=f'{len(imaging)} processed visits')
        add(rows, 'Tier 2 validated visit rate', imaging.assign(v=imaging.validated.astype(float)).groupby('patient_id').v.mean(),
            notes=f'{len(imaging)} processed visits')
    pd.DataFrame(rows).to_csv(EXPORTS / 'primary_endpoints.csv', index=False)
    pd.DataFrame(statistics).to_csv(EXPORTS / 'statistical_results.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
