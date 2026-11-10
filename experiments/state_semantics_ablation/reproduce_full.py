#!/usr/bin/env python3
"""Step 0: reproduce the archived Full OncoState (Exp 2.8) results in a fresh directory before any ablation is run.

Re-runs, with the unchanged official code, (i) the persistent synchronization of all I-SPY2 patients T0..T3, (ii) the missing-visit
stress test (595 complete patients) and (iii) the delayed-evidence test (first 100), and compares them with the archive:
version files byte-for-byte (same SHA-256), state_versions.csv, missingness_stress.csv and delayed_evidence.csv. Exit code 1 on any
discrepancy (the ablation must then not be run)."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E28 = REPO / 'experiments/exp28_repeated_synchronization'
sys.path.insert(0, str(E28))
from evaluation.continuity import state_checks  # noqa: E402
from features.measurement_features import ISPY2EvidenceSource, TP2, ispy2_cfg  # noqa: E402
from paths import EXPORTS, STATES  # noqa: E402
from state.synchronization import persistent_step  # noqa: E402
from state.versioning import VersionStore  # noqa: E402
from stress_tests.delayed_evidence import run_one as delayed_one  # noqa: E402
from stress_tests.missing_visit import run_one as missing_one  # noqa: E402

OUT = REPO / 'outputs/state_semantics_ablation/full_reproduction'


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    source = ISPY2EvidenceSource()
    patients = sorted(source.visits.patient_id.unique())
    store = VersionStore(OUT / 'patient_states')
    rows, bad_files = [], []
    for pid in patients:
        cfg = ispy2_cfg(pid)
        prev = None
        for tp in TP2:
            s, _ = persistent_step(store, source, pid, tp, cfg, now=f'protocol-{tp}')
            c = state_checks(s, prev)
            rows.append(dict(patient_id=pid, timepoint=tp, state_version=s['state_version'], parent_version=s['parent_version'],
                             n_historical_observations=len(s['history']), provenance_complete=c['provenance_complete'],
                             historical_mutation_detected=c['historical_mutation_detected'], history_retention_rate=c['history_retention_rate'],
                             lineage_integrity=True))
            prev = s
        assert store.verify(pid)
        for v in store.lineage(pid)['versions']:
            a, b = (STATES / pid / v['file']), (OUT / 'patient_states' / pid / v['file'])
            if not a.exists() or hashlib.sha256(a.read_bytes()).hexdigest() != hashlib.sha256(b.read_bytes()).hexdigest():
                bad_files.append(f'{pid}/{v["file"]}')
    new = pd.DataFrame(rows)
    arch = pd.read_csv(EXPORTS / 'state_versions.csv', dtype={'patient_id': str})
    cols = [c for c in new.columns if c in arch.columns]
    sv_ok = new[cols].reset_index(drop=True).astype(str).equals(arch[cols].reset_index(drop=True).astype(str))
    manifest = pd.read_csv(EXPORTS / 'ispy2_patient_manifest.csv', dtype={'patient_id': str})
    complete_pts = sorted(manifest.loc[manifest.complete_T0_T1_T2_T3, 'patient_id'])
    ms = VersionStore(OUT / 'stress_missing'); ds = VersionStore(OUT / 'stress_delayed')
    miss = pd.DataFrame([missing_one(source, ms, p, ispy2_cfg(p)) for p in complete_pts])
    dele = pd.DataFrame([delayed_one(source, ds, p, ispy2_cfg(p)) for p in complete_pts[:100]])
    miss.to_csv(OUT / 'missingness_stress.csv', index=False); dele.to_csv(OUT / 'delayed_evidence.csv', index=False)
    am = pd.read_csv(EXPORTS / 'missingness_stress.csv', dtype={'patient_id': str})
    ad = pd.read_csv(EXPORTS / 'delayed_evidence.csv', dtype={'patient_id': str})
    m_ok = miss.astype(str).equals(am[miss.columns].astype(str))
    d_ok = dele.astype(str).equals(ad[dele.columns].astype(str))
    rep = dict(n_patients=len(patients), n_version_files=int(len(rows)), version_files_byte_identical=not bad_files,
               n_version_file_mismatches=len(bad_files), first_mismatches=bad_files[:10], state_versions_csv_identical=bool(sv_ok),
               n_missing_visit_patients=len(miss), missingness_stress_identical=bool(m_ok),
               n_delayed_patients=len(dele), delayed_evidence_identical=bool(d_ok))
    rep['reproduced'] = bool(not bad_files and sv_ok and m_ok and d_ok)
    (OUT / 'reproduction_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep['reproduced'] else 1)


if __name__ == '__main__':
    main()
