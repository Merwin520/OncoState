"""Run the positive-control detector validation over every I-SPY2 patient's stored versions."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.detector_validation import validate_patient  # noqa: E402
from paths import EXPORTS, STATES  # noqa: E402
from state.versioning import VersionStore  # noqa: E402

SCHEDULE = ['T0', 'T1', 'T2', 'T3']


def main():
    store = VersionStore(STATES)
    rows = [dict(patient_id=p.name, **validate_patient(store.all_versions(p.name), SCHEDULE))
            for p in sorted(STATES.iterdir()) if p.is_dir()]
    d = pd.DataFrame(rows)
    d.to_csv(EXPORTS / 'detector_validation.csv', index=False)
    summary = pd.DataFrame([dict(check=c, n_patients_tested=int(d[c].notna().sum()), detected_or_correct=int(d[c].fillna(False).astype(bool).sum()))
                            for c in d.columns if c != 'patient_id'])
    summary.to_csv(EXPORTS / 'detector_validation_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
