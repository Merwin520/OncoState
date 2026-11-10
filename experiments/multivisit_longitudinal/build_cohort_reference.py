#!/usr/bin/env python3
"""Step 1 (before any system runs): cohort manifest + frozen reference trajectories, from released masks and the clinical sheet only.

Pre-specified rules (experiment_config.yaml):
  visit      = a released Timepoint_k folder; VALID if its tumorMask exists, is readable and has labels within {0..4}
  order      = by timepoint index k (dataset export labels post-treatment 1, 2, ... in acquisition order); verified against the clinical
               'days from diagnosis to k-th MRI': if >= 2 valid visits have days and they are not strictly increasing -> patient excluded
  inclusion  = >= 3 valid, objectively ordered visits; every such patient is included
Writes cohort_manifest.csv, reference_trajectories.csv, measurements_frozen.json, reference_freeze.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mu_glioma_adapter as A  # noqa: E402
import queries as Q  # noqa: E402

OUT = HERE.parents[1] / 'outputs/multivisit_longitudinal'
MIN_VISITS = 3


def patient_visits(pid):
    """Return (rows, valid_visits_in_order, exclusion_reason or None)."""
    rows, valid = [], []
    for tp in A.released_timepoints(pid):
        mods = A.available_modalities(pid, tp)
        seg = A.mask_path(pid, tp).exists()
        reason = None
        if not seg:
            reason = 'no tumorMask'
        else:
            try:
                A.measurement(pid, tp)
            except Exception as e:                       # unreadable / unexpected labels
                reason = f'mask invalid: {type(e).__name__}: {e}'
        rows.append(dict(patient_id=pid, visit_id=tp, timepoint_label_index=A.tp_index(tp), visit_date_or_relative_time=A.mri_day(pid, tp),
                         time_unit='days_from_diagnosis', available_modalities=';'.join(m.replace('brain_', '') for m in mods),
                         segmentation_available=seg, visit_valid=reason is None, visit_invalid_reason=reason))
        if reason is None:
            valid.append(tp)
    days = [(tp, A.mri_day(pid, tp)) for tp in valid]
    known = [d for _, d in days if d is not None]
    if len(known) >= 2 and any(b <= a for a, b in zip(known, known[1:])):
        return rows, valid, 'visit order conflict: clinical MRI days not strictly increasing with timepoint index'
    if len(valid) < MIN_VISITS:
        return rows, valid, f'fewer than {MIN_VISITS} valid visits ({len(valid)})'
    return rows, valid, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = A.clinical()
    pids = sorted(set(c.index) | {p.name for p in A.NIFTI.iterdir() if p.is_dir()})
    man, ref, included, pexcl = [], [], {}, {}
    for pid in pids:
        rows, valid, excl = patient_visits(pid)
        tre_ok = pid in c.index
        if not rows:
            man.append(dict(patient_id=pid, visit_id=None, visit_index=None, visit_date_or_relative_time=None, available_modalities=None,
                            segmentation_available=False, treatment_metadata_available=tre_ok, included_or_excluded='excluded',
                            exclusion_reason='no released imaging timepoint'))
            pexcl[pid] = 'no released imaging timepoint'
            continue
        pexcl[pid] = excl
        _, bad_days = A.treatment_events(pid)
        for r in rows:
            vi = valid.index(r['visit_id']) + 1 if r['visit_id'] in valid else None
            r.update(visit_index=vi, treatment_metadata_available=tre_ok and r['visit_date_or_relative_time'] is not None,
                     unparseable_treatment_days=json.dumps(bad_days) if bad_days else None,
                     included_or_excluded='included' if excl is None and r['visit_valid'] else 'excluded',
                     exclusion_reason=excl if excl else (None if r['visit_valid'] else r['visit_invalid_reason']))
            man.append(r)
        if excl is None:
            included[pid] = valid
    for pid, visits in included.items():
        reports = {v: A.visit_record(pid, v, visits)[0]['treatments_reported_since_previous_visit'] for v in visits}
        seq = []
        for i, v in enumerate(visits):
            rec, _ = A.visit_record(pid, v, visits)
            seq.append((v, rec['measurements']['burden_ml']))
            ans = Q.reference_answers(seq, reports)
            b = [x[1] for x in seq]
            ref.append(dict(patient_id=pid, visit_id=v, visit_depth=i + 1, days_from_diagnosis=rec['days_from_diagnosis'], n_prior_visits=i,
                            burden_ml=b[-1], **{k: rec['measurements'][k] for k in ('et_ml', 'netc_ml', 'snfh_ml', 'resection_cavity_ml')},
                            previous_visit=ans.get('Q2', {}).get('previous_visit'), previous_burden_ml=ans.get('Q2', {}).get('previous_burden_ml'),
                            change_from_previous_ml=ans.get('Q3', {}).get('delta_ml'), pct_change_from_previous=ans.get('Q3', {}).get('pct_change'),
                            first_visit=visits[0], change_from_first_ml=ans.get('Q4', {}).get('delta_ml'), pct_change_from_first=ans.get('Q4', {}).get('pct_change'),
                            nadir_visit=ans.get('Q5', {}).get('nadir_visit'), nadir_burden_ml=ans.get('Q5', {}).get('nadir_burden_ml', b[-1]),
                            change_from_nadir_ml=ans.get('Q5', {}).get('current_minus_nadir_ml'),
                            step_direction=ans.get('Q6', {}).get('current_step'), reversal=ans.get('Q6', {}).get('reversal'),
                            treatments_reported_this_visit=json.dumps(reports[v]) if reports[v] is not None else None,
                            treatments_started_by_visit=json.dumps(ans['Q7']['treatments_started']) if 'Q7' in ans else None,
                            evaluable_queries=';'.join(sorted(ans)), reference_answers=Q.dumps(ans)))
    m = pd.DataFrame(man)
    cols = ['patient_id', 'visit_id', 'visit_index', 'visit_date_or_relative_time', 'available_modalities', 'segmentation_available',
            'treatment_metadata_available', 'included_or_excluded', 'exclusion_reason']
    m = m[cols + [k for k in m.columns if k not in cols]]
    m.to_csv(OUT / 'cohort_manifest.csv', index=False)
    r = pd.DataFrame(ref)
    r.to_csv(OUT / 'reference_trajectories.csv', index=False)
    A.save_measure_cache(OUT / 'measurements_frozen.json')
    nv = pd.Series({p: len(v) for p, v in included.items()})
    allv = m[m.visit_valid == True].groupby('patient_id').size()  # noqa: E712
    counts = dict(patients_in_release=int(m.patient_id.nunique()), released_visits=int(m.visit_id.notna().sum()),
                  patients_ge3_valid=int((allv >= 3).sum()), patients_ge4_valid=int((allv >= 4).sum()), patients_ge5_valid=int((allv >= 5).sum()),
                  max_valid_visits=int(allv.max()), included_patients=len(included), included_visits=int(nv.sum()),
                  included_ge4=int((nv >= 4).sum()), included_ge5=int((nv >= 5).sum()), included_ge6=int((nv >= 6).sum()),
                  visits_per_included_patient=nv.value_counts().sort_index().to_dict(),
                  patient_exclusion_reasons=pd.Series({p: e for p, e in pexcl.items() if e}).value_counts().to_dict(),
                  invalid_visits=m[m.visit_valid == False].visit_invalid_reason.value_counts().to_dict())  # noqa: E712
    freeze = dict(counts=counts, reference_sha256=A.sha256(OUT / 'reference_trajectories.csv'), manifest_sha256=A.sha256(OUT / 'cohort_manifest.csv'),
                  measurements_sha256=A.sha256(OUT / 'measurements_frozen.json'), measurement_version=A.MEASUREMENT_VERSION)
    (OUT / 'reference_freeze.json').write_text(json.dumps(freeze, indent=1, default=str))
    print(json.dumps(freeze, indent=1, default=str))


if __name__ == '__main__':
    main()
