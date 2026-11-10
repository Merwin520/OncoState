"""Experiment 2.3-A: real HCUCH longitudinal target-lesion response reconstruction.

Only target-lesion measurements exist in the dataset (no new-lesion or non-target information, no expert response labels), so the
output is TARGET-LESION response; overall RECIST response is never assigned (the pipeline abstains on it). The reference is the
independent RECIST 1.1 specification in exp23_rules.spec_target applied to the expert measurements.

Non-circular check: every target lesion is re-measured from its segmentation mask (long axis for tumors, short axis for lymph
nodes) and the response derived from those state-side measurements is compared with the response derived from expert measurements.
"""
import os
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import ConvexHull

import exp23_rules as RL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
DATA = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'HCUCH RECIST CT/final-formatted/final-formatted'))
NODE_MIN_BASELINE, TUMOR_MIN_BASELINE = 15.0, 10.0


def hull_pts(pts):
    if len(pts) < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:
        return pts


def axes_mm(mask2d, sp_xy):
    """Long axis and perpendicular width (mm) of a 2D binary mask (convex hull), pixel size added as in the 2.2 pipeline."""
    pts = np.argwhere(mask2d).astype(float) * np.array([sp_xy[1], sp_xy[0]])       # (row->y, col->x) in mm
    if len(pts) == 0:
        return 0.0, 0.0
    h = hull_pts(pts)
    d = np.linalg.norm(h[:, None] - h[None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    L = d[i, j]
    if L == 0:
        return float(min(sp_xy)), float(min(sp_xy))
    u = (h[j] - h[i]) / L; w = np.array([-u[1], u[0]])
    proj = h @ w
    return float(L + min(sp_xy)), float(proj.max() - proj.min() + min(sp_xy))


def measure_from_mask(mask_path, label_value, kind):
    img = sitk.ReadImage(str(mask_path)); arr = sitk.GetArrayFromImage(img); sp = img.GetSpacing()
    m = arr == label_value
    best_L, best_W = 0.0, 0.0
    for z in np.where(m.reshape(m.shape[0], -1).any(1))[0]:
        L, W = axes_mm(m[z], (sp[0], sp[1]))
        best_L, best_W = max(best_L, L), max(best_W, W)
    return best_W if kind == 'n' else best_L, best_L, best_W


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(DATA / 'recist_measurements.csv')
    pat = pd.read_csv(DATA / 'metadata/patients.csv').set_index('patient_id')
    # lesion type / organ from the label files
    kinds, organs, masks = [], [], []
    for _, r in m.iterrows():
        sub = 'train' if str(r.subset).startswith('train') else 'test'
        lab = json.loads((DATA / f'images/{r.region}/{sub}/labels/{r.filename.replace(".nii.gz", ".json")}').read_text())
        k, o = lab[str(r.lesion_label_value)].split(',', 1)
        kinds.append(k); organs.append(o); masks.append(str(DATA / f'images/{r.region}/{sub}/masks/{r.filename}'))
    m['kind'], m['organ'], m['mask_path'] = kinds, organs, masks
    print('measuring', len(m), 'target lesions from masks ...', flush=True)
    mm = [measure_from_mask(r.mask_path, int(r.lesion_label_value), r.kind) for _, r in m.iterrows()]
    m['mask_measure_mm'] = [x[0] for x in mm]; m['mask_long_mm'] = [x[1] for x in mm]; m['mask_short_mm'] = [x[2] for x in mm]
    m['study_date'] = m.study_date.astype(str)
    m.drop(columns=['mask_path']).to_csv(OUT / 'real_hcuch_measurements.csv', index=False)

    # per patient-study sums (across series of the same study)
    st = (m.groupby(['patient_id', 'study_uuid', 'study_date', 'study_order'])
           .agg(n=('lesion_label_alias', 'count'), aliases=('lesion_label_alias', lambda s: ''.join(sorted(s))),
                expert_sum=('recist_measurement_mm', 'sum'), mask_sum=('mask_measure_mm', 'sum')).reset_index().sort_values(['patient_id', 'study_date']))
    states, rows = {}, []
    applic = {'baseline_below_min_size': 0, 'baseline_lesions': 0, 'patients_gt2_per_organ': 0, 'patients_gt5_targets': 0}
    for pid, g in st.groupby('patient_id'):
        g = g.reset_index(drop=True)
        les = m[m.patient_id == pid]
        states[int(pid)] = {'patient_id': int(pid), 'diagnosis': pat.loc[pid, 'diagnosis'] if pid in pat.index else None,
                            'studies': [{'study_order': r.study_order, 'date': r.study_date, 'sum_mm': float(r.expert_sum), 'n_targets': int(r.n)} for r in g.itertuples()],
                            'target_lesions': [{'alias': a, 'type': d.kind.iloc[0], 'organ': d.organ.iloc[0],
                                                'measurements': [{'date': x.study_date, 'study_order': x.study_order, 'mm': float(x.recist_measurement_mm), 'source': 'expert'} for x in d.itertuples()]}
                                               for a, d in les.groupby('lesion_label_alias')]}
        base = les[les.study_order == 'baseline']
        applic['baseline_lesions'] += len(base)
        applic['baseline_below_min_size'] += int(((base.kind == 'n') & (base.recist_measurement_mm < NODE_MIN_BASELINE)).sum() + ((base.kind != 'n') & (base.recist_measurement_mm < TUMOR_MIN_BASELINE)).sum())
        applic['patients_gt2_per_organ'] += int((base.groupby('organ').size() > 2).any()); applic['patients_gt5_targets'] += int(len(base) > 5)
        if len(g) < 2 or g.study_order.iloc[0] != 'baseline':
            continue
        base_alias = g.aliases.iloc[0]
        for k in range(1, len(g)):
            r = g.iloc[k]
            complete = (r.aliases == base_alias)
            hist_e = [float(x) for x in g.expert_sum.iloc[:k]]; hist_m = [float(x) for x in g.mask_sum.iloc[:k]]
            case = {'history': hist_e, 'current_sum': float(r.expert_sum), 'new_lesions': None, 'non_target': None, 'lesions_missing': not complete}
            eng = RL.engine_response(case) if complete else {'response': 'NE', 'rule_ids': ['POLICY.ABSTAIN'], 'abstained': True, 'target_response': None, 'trace': {}}
            ref_t, ref_id = RL.spec_target(hist_e, r.expert_sum) if complete else ('NE', 'POLICY.ABSTAIN')
            mask_t, _ = RL.spec_target(hist_m, r.mask_sum) if complete else ('NE', '')
            d = RL.derived(case)
            ref_overall = RL.spec_response(case)[0] if complete else 'NE'
            rows.append(dict(patient_id=int(pid), diagnosis=pat.loc[pid, 'diagnosis'] if pid in pat.index else '', study_pair=f'{g.study_order.iloc[0]}->{r.study_order}',
                             baseline_study_date=g.study_date.iloc[0], followup_study_date=r.study_date, n_target_lesions=int(r.n), baseline_sum=d.get('baseline_sum'),
                             prior_nadir_sum=d.get('nadir_sum'), followup_sum=float(r.expert_sum), change_from_baseline=d.get('change_baseline_pct'), change_from_nadir=d.get('change_nadir_pct'),
                             absolute_change_from_nadir_mm=d.get('absolute_change_from_nadir_mm'),
                             new_lesion_status='not recorded', non_target_status='not recorded', evidence_complete_target=bool(complete), evidence_complete_overall=False,
                             applicable_guideline='RECIST 1.1', reference_response_component='target-lesion', reference_response=ref_t,
                             predicted_response=eng['target_response'] if eng['target_response'] else 'NE', predicted_overall_response=eng['response'],
                             reference_overall_response=ref_overall, overall_correct=(eng['response'] == ref_overall), correct=(eng['target_response'] if eng['target_response'] else 'NE') == ref_t,
                             rule_ids_used=';'.join(eng['rule_ids']), reference_rule_id=ref_id, rule_correct=(ref_id in eng['rule_ids']), guideline_version='RECIST 1.1 (Eisenhauer 2009)',
                             traceability_complete=bool(eng['trace'].get('steps')) and all('rule_id' in s for s in eng['trace']['steps'] if s['step'] != 'derive'),
                             mask_derived_followup_sum=float(r.mask_sum), mask_derived_baseline_sum=hist_m[0], mask_derived_response=mask_t,
                             mask_response_agrees_with_expert=(mask_t == ref_t)))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'real_hcuch_cases.csv', index=False)
    (OUT / 'real_patient_states.json').write_text(json.dumps(states, indent=1))
    info = dict(n_measurement_rows=len(m), n_patients_with_measurements=int(m.patient_id.nunique()), n_studies_with_measurements=int(m.study_uuid.nunique()),
                n_series_with_measurements=int(m.uuid.nunique()), n_patients_evaluable=int(df.patient_id.nunique()) if len(df) else 0, n_real_cases=len(df),
                n_target_lesion_response_only=int((df.reference_overall_response == 'NE').sum()), n_overall_response_assignable=int((df.reference_overall_response != 'NE').sum()), rule_applicability=applic,
                mask_vs_expert=dict(n=len(m), mae_mm=float((m.mask_measure_mm - m.recist_measurement_mm).abs().mean()), bias_mm=float((m.mask_measure_mm - m.recist_measurement_mm).mean()),
                                    mae_tumor_mm=float((m[m.kind != 'n'].mask_measure_mm - m[m.kind != 'n'].recist_measurement_mm).abs().mean()),
                                    mae_node_mm=float((m[m.kind == 'n'].mask_measure_mm - m[m.kind == 'n'].recist_measurement_mm).abs().mean())))
    (OUT / 'real_summary.json').write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, indent=1, default=float)); print(df[['patient_id', 'study_pair', 'baseline_sum', 'prior_nadir_sum', 'followup_sum', 'change_from_baseline', 'change_from_nadir', 'reference_response', 'predicted_response', 'mask_derived_response']].round(1).to_string())


if __name__ == '__main__':
    main()
