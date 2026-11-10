"""Run missing-visit and secondary delayed-evidence tests on audited complete patients."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.measurement_features import ISPY2EvidenceSource, ispy2_cfg  # noqa: E402
from paths import EXPORTS, work_dir  # noqa: E402
from state.versioning import VersionStore  # noqa: E402
from stress_tests.delayed_evidence import run_one as delayed_one, verify as delayed_verify  # noqa: E402
from stress_tests.missing_visit import run_one as missing_one, verify as missing_verify  # noqa: E402


def main():
    source = ISPY2EvidenceSource()
    manifest = pd.read_csv(EXPORTS / 'ispy2_patient_manifest.csv', dtype={'patient_id': str})
    patients = sorted(manifest.loc[manifest.complete_T0_T1_T2_T3, 'patient_id'])
    missing_store = VersionStore(work_dir() / 'stress_states_missing')
    missing_rows = []
    for number, patient_id in enumerate(patients, 1):
        if len(missing_store.lineage(patient_id)['versions']) == 5:   # resume: re-verify from the stored versions, never assume
            missing_rows.append(missing_verify(missing_store, patient_id))
        else:
            missing_rows.append(missing_one(source, missing_store, patient_id, ispy2_cfg(patient_id)))
        if number % 100 == 0:
            print(f'missing-visit stress {number}/{len(patients)}', flush=True)
    pd.DataFrame(missing_rows).to_csv(EXPORTS / 'missingness_stress.csv', index=False)
    delayed_store = VersionStore(work_dir() / 'stress_states_delayed')
    delayed_rows = []
    for patient_id in patients[:100]:
        if len(delayed_store.lineage(patient_id)['versions']) == 4:   # resume: re-verify from the stored versions
            delayed_rows.append(delayed_verify(delayed_store, patient_id, ispy2_cfg(patient_id)))
        else:
            delayed_rows.append(delayed_one(source, delayed_store, patient_id, ispy2_cfg(patient_id)))
    pd.DataFrame(delayed_rows).to_csv(EXPORTS / 'delayed_evidence.csv', index=False)
    print(f'wrote {len(missing_rows)} missing-visit and {len(delayed_rows)} delayed-evidence results')


if __name__ == '__main__':
    main()
