#!/usr/bin/env python3
"""Stage 1 data audit (no model is fitted). Writes results/audit_patient_table.csv and results/audit_counts.json."""
import json
import numpy as np
import pandas as pd
from common import ISPY1, ISPY2, RES, T1, T2, _f


def main():
    I = ISPY2(); rows = []
    for p in I.pts:
        c = I.clinical(p); r = dict(patient_id=p, arm=c['arm'], pcr=I.clin.at[p, 'pCR'] if p in I.clin.index else np.nan,
                                    HR=c['HR'], HER2=c['HER2'], MP=c['MP'], age=c['age'], postmeno=c['postmeno'])
        for tp in T2:
            o = I.src.observation(p, tp)
            r[f'{tp}_mri'] = o is not None
            r[f'{tp}_day'] = o.get('study_day_offset') if o else np.nan
            for key, n, _ in __import__('common').M2:
                r[f'{tp}_{n}'] = _f(o, key) if o else np.nan
        rows.append(r)
    A = pd.DataFrame(rows); A.to_csv(RES / 'audit_patient_table.csv', index=False)
    obs = lambda tp: A[f'{tp}_mri'] & A[f'{tp}_ftv_pe'].notna()  # noqa: E731
    cnt = dict(ispy2_patients=len(A), pcr_available=int(A.pcr.notna().sum()), pcr_prevalence=float(A.pcr.mean()),
               arms=int(A.arm.nunique()), arm_missing=int(A.arm.isna().sum()),
               mri_by_visit={tp: int(A[f'{tp}_mri'].sum()) for tp in T2}, ftv_by_visit={tp: int(obs(tp).sum()) for tp in T2},
               ftv_zero_by_visit={tp: int((A[f'{tp}_ftv_pe'] == 0).sum()) for tp in T2},
               day_available_by_visit={tp: int(A[f'{tp}_day'].notna().sum()) for tp in T2},
               day_median_by_visit={tp: float(A[f'{tp}_day'].median()) for tp in T2},
               non_increasing_days=int(sum((np.diff(r.dropna().values) <= 0).any() for _, r in A[[f'{t}_day' for t in T2]].iterrows())),
               clinical_missing={c: int(A[c].isna().sum()) for c in ('HR', 'HER2', 'MP', 'age', 'postmeno')},
               t0_t1_ftv=int((obs('T0') & obs('T1')).sum()))
    pairs = {f'{a}->{b}': int((obs(a) & obs(b)).sum()) for a, b in (('T0', 'T1'), ('T1', 'T2'), ('T2', 'T3'), ('T0', 'T2'), ('T0', 'T3'), ('T1', 'T3'))}
    cnt['ftv_transitions'] = pairs
    cnt['all_four_visits'] = int((obs('T0') & obs('T1') & obs('T2') & obs('T3')).sum())
    cnt['revision_eligible_T1_observed'] = int(obs('T1').sum())
    cnt['revision_with_T2_and_T3'] = int((obs('T1') & obs('T2') & obs('T3')).sum())
    cnt['late_arrival_eligible_T1_and_T2'] = int((obs('T1') & obs('T2')).sum())
    E = ISPY1(); d = E.src.data
    cnt['ispy1'] = dict(patients=len(d), pcr_missing=int(d.PCR.isna().sum()), baseline_ld_missing_among_pcr=int((d.PCR.notna() & d['MRI LD Baseline'].isna()).sum()),
                        analysed=len(E.pts), ld_by_visit={tp: int(d.loc[E.pts, c].notna().sum()) for tp, c in zip(T2, ['MRI LD Baseline', 'MRI LD 1-3dAC', 'MRI LD InterReg', 'MRI LD PreSurg'])},
                        ld_transitions={f'{a}->{b}': int((d.loc[E.pts, ca].notna() & d.loc[E.pts, cb].notna()).sum()) for (a, ca), (b, cb) in
                                        ((('T0', 'MRI LD Baseline'), ('T1', 'MRI LD 1-3dAC')), (('T1', 'MRI LD 1-3dAC'), ('T2', 'MRI LD InterReg')),
                                         (('T2', 'MRI LD InterReg'), ('T3', 'MRI LD PreSurg')))},
                        ftv_available=False, treatment_arm_available=False)
    ld2 = {f'{a}->{b}': int((A[f'{a}_ld'].notna() & A[f'{b}_ld'].notna()).sum()) for a, b in (('T0', 'T1'), ('T1', 'T2'), ('T2', 'T3'))}
    cnt['ispy2_ld_transitions'] = ld2
    (RES / 'audit_counts.json').write_text(json.dumps(cnt, indent=1, default=float))
    print(json.dumps(cnt, indent=1, default=float))
    print(A.arm.value_counts().to_string())


if __name__ == '__main__':
    main()
