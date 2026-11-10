"""Experiment 2.7 Phases 1-6: observed pre-TACE -> TACE -> post-TACE transition audit.

Inputs  : DICOM index run (index_dicom.py), official TCIA clinical XLSX (hash-checked), repository RECIST rule (exp23_rules.spec_target).
Outputs : <run>/transition_manifest.csv, imaging_selection.csv, segmentation_manifest.csv, treatment_action_audit.csv,
          response_mapping.csv, inter_reader_agreement.csv, propensity_overlap.csv, cohort_flow.csv, audit_summary.json.
No model is trained. No source file is modified. Every derived value keeps its source column / DICOM UID.

Usage: audit_transitions.py <dicom_index_run_dir> <output_run_dir>
"""
import os
import hashlib
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments/exp2_guideline_reasoning'))
from exp23_rules import spec_target  # noqa: E402  repository RECIST 1.1 target-response rule

CLINICAL = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'HCC-TACE-Seg_clinical_data-V2 (1).xlsx'))
CLINICAL_SHA256 = 'e9d2683dc907819aa08e4027e085854f46464bec834a09e80fc83b19a89e092b'  # == TCIA HCC-TACE-Seg_clinical_data-V2.xlsx
READERS = (1, 2, 3)
CODE = {1: 'CR', 2: 'PR', 3: 'SD', 4: 'PD'}  # verified empirically below for RECIST and mRECIST; NOT used for EASL
# Fields that describe the future or the outcome; they must never enter S_t.
FUTURE_FIELDS = ['Interval_FU', 'TTP', 'Death_1_StillAliveorLostToFU_0', 'Censored_0_progressed_1', 'OS', '1_EASL_FU', '1_EASL'] + \
    [f'{r}_{k}{s}' for k in ('RECIST', 'mRECIST') for r in READERS for s in ('_FU', '')]
BASELINE_CLINICAL = ['Interval_BL', 'hepatitis', 'age', 'agegp', 'Sex', 'Smoking', 'Alcohol', 'fhx_can', 'fhx_livc', 'Diabetes',
                     'Personal history of cancer', 'Evidence_of_cirh', 'Pathology', 'PS_bclc_0_0_1-2_1_3-4_3', 'CPS', 'Tr_Size',
                     'tumor_nodul', 'Vascular invasion', 'Metastasis', 'Lymphnodes', 'Portal Vein Thrombosis', 'T_involvment', 'AFP',
                     'AFP_group', 'CLIP_Score', 'CLIP', 'Okuda', 'TNM', 'BCLC']


# ---------------------------------------------------------------- response rules
def recist_like(bl, fu):
    """Target response from baseline and follow-up sums; with only two timepoints the nadir is the baseline."""
    if pd.isna(bl) or pd.isna(fu):
        return None
    if fu == 0:
        return 'CR'  # all target lesions (mRECIST: viable enhancing tumor) gone
    return spec_target([bl], fu)[0]


def majority(codes):
    """Category chosen by >= 2 of the readers who read the case; None when no category reaches 2 votes."""
    vals = pd.Series([x for x in codes if pd.notna(x)]).value_counts()
    return CODE[int(vals.index[0])] if len(vals) and vals.iloc[0] >= 2 else None


def easl_like(bl, fu):
    """EASL (bidimensional product of enhancing tumor): CR 0, PR <= -50 %, PD >= +25 %."""
    if pd.isna(bl) or pd.isna(fu):
        return None
    if fu == 0:
        return 'CR'
    ch = (fu - bl) / bl
    return 'PD' if ch >= .25 else 'PR' if ch <= -.5 else 'SD'


# ---------------------------------------------------------------- treatment normalization
AGENTS = {'cisplatin': ('cisplatin', 'cisplastin'), 'doxorubicin': ('doxorubicin',), 'mitomycin_c': ('mitomycin',)}


def normalize_treatment(raw):
    if pd.isna(raw):
        return dict(treatment_normalized='UNKNOWN', agents='', embolic_material='', dose='not recorded',
                    normalization_rule='missing in source; not imputed')
    low = raw.lower()
    agents = [a for a, keys in AGENTS.items() if any(k in low for k in keys)]
    if 'bead' in low:
        return dict(treatment_normalized='DEB-TACE', agents=';'.join(agents), embolic_material='LC Bead (drug-eluting)', dose='not recorded',
                    normalization_rule="'beads' in raw string -> drug-eluting-bead TACE")
    rule = 'chemotherapy agents without beads -> conventional TACE (cTACE)'
    if 'cisplastin' in low:
        rule += "; 'Cisplastin' read as cisplatin (spelling)"
    return dict(treatment_normalized='cTACE', agents=';'.join(agents), embolic_material='not recorded in source', dose='not recorded',
                normalization_rule=rule)


# ---------------------------------------------------------------- helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def norm_desc(d):
    return '' if pd.isna(d) else re.sub(r'RECON \d+:\s*', '', str(d).upper()).strip()


def recon_no(d):
    m = None if pd.isna(d) else re.search(r'Recon (\d+)', str(d))
    return m.group(1) if m else None


def phase_from_metadata(desc):
    d = '' if pd.isna(desc) else desc.upper()
    if 'PRE' in d:
        return 'non-contrast'
    if 'ARTERIAL' in d:
        return 'arterial (series description)'
    if 'VENOUS' in d:
        return 'venous (series description)'
    if not d:
        return 'unknown (no description)'
    return 'contrast-enhanced; arterial/portal-venous not determinable from metadata'


def smd(a, b):
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def main():
    idx_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    assert sha256(CLINICAL) == CLINICAL_SHA256, 'clinical XLSX differs from the verified TCIA file'
    c = pd.read_excel(CLINICAL, sheet_name='data table')
    s = pd.read_csv(idx_dir / 'series_index.csv', dtype={'StudyDate': str})
    g = pd.read_csv(idx_dir / 'seg_index.csv', dtype={'seg_study_date': str})
    ct = s[s.Modality == 'CT'].copy()
    assert set(c.TCIA_ID) == set(ct.PatientID) and c.TCIA_ID.is_unique
    summary = {'clinical_file': str(CLINICAL), 'clinical_sha256': CLINICAL_SHA256, 'dicom_index_run': str(idx_dir)}

    # ---------------- timepoints from DICOM study dates (independent of the clinical table)
    ct['StudyDate_dt'] = pd.to_datetime(ct.StudyDate, format='%Y%m%d')
    studies = ct.groupby(['PatientID', 'StudyInstanceUID']).StudyDate_dt.first().reset_index().sort_values(['PatientID', 'StudyDate_dt'])
    n_studies = studies.groupby('PatientID').size()
    summary['ct_studies_per_patient'] = n_studies.value_counts().to_dict()
    tp = studies.groupby('PatientID').agg(bl_study=('StudyInstanceUID', 'first'), bl_date=('StudyDate_dt', 'first'),
                                          fu_study=('StudyInstanceUID', 'last'), fu_date=('StudyDate_dt', 'last'))
    tp['same_day'] = tp.bl_date == tp.fu_date
    ct['timepoint'] = np.where(ct.StudyInstanceUID.isin(tp.bl_study), 'BL', 'FU')

    # ---------------- segmentation manifest
    segs = []
    for uid, gg in g.groupby('seg_series_uid'):
        r0 = gg.iloc[0]
        p = r0.PatientID
        ref, method = r0.referenced_series_uids, 'ReferencedSeriesSequence'
        if pd.isna(ref):  # resolve by exact slice-position coverage within the patient's CT series (reported, flagged)
            ref, method = resolve_seg_by_geometry(s, p, uid), 'slice-position match (no ReferencedSeriesSequence)'
        refrow = s.set_index('SeriesInstanceUID').loc[ref] if ref in set(s.SeriesInstanceUID) else None
        lab = gg.set_index('segment_label')
        mass = lab.loc['Mass']
        comps = json.loads(mass.component_voxels)
        vv = mass.voxel_volume_mm3
        seg_tp = 'BL' if refrow is not None and refrow.StudyInstanceUID == tp.loc[p, 'bl_study'] else ('FU' if refrow is not None else 'unresolved')
        segs.append(dict(
            patient_id=p, seg_series_uid=uid, seg_study_uid=r0.seg_study_uid, seg_study_date=r0.seg_study_date,
            referenced_ct_series_uid=ref, reference_method=method, referenced_series_description=None if refrow is None else refrow.SeriesDescription,
            timepoint=seg_tp, segments=';'.join(gg.segment_label), algorithm_type=';'.join(sorted(set(gg.algorithm_type))),
            has_liver='Liver' in lab.index, has_mass='Mass' in lab.index, has_portal_vein='Portal vein' in lab.index,
            has_aorta='Abdominal aorta' in lab.index, has_necrosis='Necrosis' in lab.index,
            mass_volume_ml=mass.voxels * vv / 1000, mass_components_3d=int(mass.n_components_3d),
            mass_largest_component_ml=(comps[0] * vv / 1000) if comps else 0.0,
            liver_volume_ml=lab.loc['Liver'].voxels * lab.loc['Liver'].voxel_volume_mm3 / 1000 if 'Liver' in lab.index else np.nan,
            is_baseline_only=seg_tp == 'BL', is_followup=seg_tp == 'FU', multiple_mass_components=int(mass.n_components_3d) > 1,
            quality_flags='seg_reference_resolved_by_geometry;frame_of_reference_uid_mismatch;in_plane_origin_mismatch' if method != 'ReferencedSeriesSequence' else ''))
    seg = pd.DataFrame(segs).sort_values('patient_id')
    seg['mass_to_liver_ratio'] = seg.mass_volume_ml / seg.liver_volume_ml
    seg.to_csv(out / 'segmentation_manifest.csv', index=False)
    summary['seg'] = dict(n=len(seg), timepoints=seg.timepoint.value_counts().to_dict(), segments=seg.segments.value_counts().to_dict(),
                          multiple_mass_components=int(seg.multiple_mass_components.sum()),
                          resolved_by_geometry=seg[seg.reference_method != 'ReferencedSeriesSequence'].patient_id.tolist())
    segp = seg.set_index('patient_id')

    # ---------------- imaging selection (deterministic)
    sel = []
    for p in sorted(tp.index):
        bl_ref = segp.loc[p, 'referenced_ct_series_uid']
        b = s.set_index('SeriesInstanceUID').loc[bl_ref]
        sel.append(sel_row(p, 'BL', b, bl_ref, 'official tumor SEG references this series', segp.loc[p, 'seg_series_uid'], segp.loc[p, 'quality_flags']))
        cand = ct[(ct.PatientID == p) & (ct.StudyInstanceUID == tp.loc[p, 'fu_study'])].copy()
        cand['orig'] = cand.ImageType.fillna('').str.startswith('ORIGINAL')
        cand['enh'] = ~cand.SeriesDescription.fillna('').str.upper().str.contains('PRE') & cand.SeriesDescription.notna()
        cand['same_protocol_recon'] = [norm_desc(d) == norm_desc(b.SeriesDescription) and recon_no(d) == recon_no(b.SeriesDescription)
                                       for d in cand.SeriesDescription]
        cand['thk'] = cand.SliceThickness.astype(float)
        cand = cand.sort_values(['orig', 'enh', 'same_protocol_recon', 'thk', 'n_files', 'SeriesNumber'],
                                ascending=[False, False, False, True, False, True])
        f = cand.iloc[0]
        why = ('ORIGINAL' if f.orig else 'no ORIGINAL series') + ', ' + ('contrast-enhanced' if f.enh else 'no enhanced series') + \
            (', same protocol/recon as BL' if f.same_protocol_recon else ', different protocol/recon from BL') + ', thinnest slice'
        flags = ';'.join(x for x, bad in (('fu_no_original_series', not f.orig), ('fu_no_enhanced_series', not f.enh),
                                              ('fu_recon_differs_from_bl', not f.same_protocol_recon),
                                              ('fu_bl_slice_thickness_differs', f.thk != float(b.SliceThickness))) if bad)
        sel.append(sel_row(p, 'FU', f, f.SeriesInstanceUID, why + ' (no follow-up SEG exists)', '', flags))
    sel = pd.DataFrame(sel)
    sel.to_csv(out / 'imaging_selection.csv', index=False)

    # ---------------- treatment audit
    tr = c.chemotherapy.apply(normalize_treatment).apply(pd.Series)
    tr.insert(0, 'treatment_raw', c.chemotherapy)
    tra = tr.fillna({'treatment_raw': '<missing>'}).groupby(['treatment_raw', 'treatment_normalized', 'agents', 'embolic_material', 'dose',
                                                               'normalization_rule'], dropna=False).size().rename('n_patients').reset_index()
    tra.to_csv(out / 'treatment_action_audit.csv', index=False)

    # ---------------- response mapping (code vs measurement-derived category)
    rm, derived = [], {}
    for crit, fn in (('RECIST', recist_like), ('mRECIST', recist_like), ('EASL', easl_like)):
        for r in (READERS if crit != 'EASL' else (1,)):
            col = f'{r}_{crit}'
            der = pd.Series([fn(b_, u_) for b_, u_ in zip(c[col + '_BL'], c[col + '_FU'])], index=c.index)
            derived[col] = der
            for code, grp in c.groupby(col, dropna=False):
                dd = der[grp.index].value_counts(dropna=False)
                for cat, n in dd.items():
                    rm.append(dict(column=col, criterion=crit, reader=r, raw_code=code, derived_category_from_measurements=cat, n=int(n)))
    rm = pd.DataFrame(rm)
    ver = {}
    for col, gg in rm.groupby('column'):
        ok = gg.dropna(subset=['raw_code'])
        if gg.criterion.iloc[0] == 'EASL':
            ver[col] = dict(mapping='NOT VALIDATED', note='only codes 3/4 occur: 3 <=> follow-up measurement 0; 4 covers PR, SD and PD by '
                            'measurement. Code meaning is undocumented; EASL category is not used as a label.')
        else:
            agree = ok[ok.raw_code.map(CODE) == ok.derived_category_from_measurements].n.sum()
            ver[col] = dict(mapping='1=CR,2=PR,3=SD,4=PD', agreement=f'{agree}/{ok.n.sum()}', verified=bool(agree == ok.n.sum()))
    rm['verified_mapping'] = rm.column.map(lambda k: ver[k]['mapping'])
    rm['mapped_category'] = [CODE.get(int(x)) if v != 'NOT VALIDATED' and pd.notna(x) else None for x, v in zip(rm.raw_code, rm.verified_mapping)]
    rm.to_csv(out / 'response_mapping.csv', index=False)
    summary['response_mapping'] = ver

    # ---------------- inter-reader agreement + consensus
    ira = []
    for crit in ('RECIST', 'mRECIST'):
        for a, b_ in combinations(READERS, 2):
            ca, cb = c[f'{a}_{crit}'], c[f'{b_}_{crit}']
            ok = ca.notna() & cb.notna()
            ira.append(dict(criterion=crit, pair=f'R{a}-R{b_}', n=int(ok.sum()), category_agreement=float((ca[ok] == cb[ok]).mean()),
                            cohen_kappa=float(cohen_kappa_score(ca[ok], cb[ok])),
                            weighted_kappa_linear=float(cohen_kappa_score(ca[ok], cb[ok], weights='linear')),
                            bl_median_abs_diff_mm=float((c[f'{a}_{crit}_BL'] - c[f'{b_}_{crit}_BL']).abs().median()),
                            fu_median_abs_diff_mm=float((c[f'{a}_{crit}_FU'] - c[f'{b_}_{crit}_FU']).abs().median())))
        codes = c[[f'{r}_{crit}' for r in READERS]]
        c[f'{crit}_n_readers'] = codes.notna().sum(axis=1)
        c[f'{crit}_all_agree'] = codes.nunique(1).eq(1) & c[f'{crit}_n_readers'].eq(3)
        c[f'{crit}_majority'] = [majority(row) for row in codes.itertuples(index=False)]
        c[f'{crit}_BL_median'] = c[[f'{r}_{crit}_BL' for r in READERS]].median(axis=1)
        c[f'{crit}_FU_median'] = c[[f'{r}_{crit}_FU' for r in READERS]].median(axis=1)
        c[f'{crit}_median_derived'] = [recist_like(x, y) for x, y in zip(c[f'{crit}_BL_median'], c[f'{crit}_FU_median'])]
        both = c[f'{crit}_majority'].notna() & c[f'{crit}_median_derived'].notna()
        ira.append(dict(criterion=crit, pair='majority vs median-measurement rule', n=int(both.sum()),
                        category_agreement=float((c[f'{crit}_majority'][both] == c[f'{crit}_median_derived'][both]).mean()),
                        note=f"all 3 readers agree: {int(c[f'{crit}_all_agree'].sum())}; no majority: "
                             f"{int((c[f'{crit}_majority'].isna() & c[f'{crit}_n_readers'].gt(0)).sum())}"))
    pd.DataFrame(ira).to_csv(out / 'inter_reader_agreement.csv', index=False)

    # ---------------- Interval_FU cross-check against imaging
    img_days = c.TCIA_ID.map((tp.fu_date - tp.bl_date).dt.days)
    fu_days = c.Interval_FU * 7
    c['interval_fu_matches_imaging'] = (fu_days - img_days).abs() <= 1.5
    c['interval_fu_matches_next_patient_imaging'] = (fu_days - img_days.shift(-1)).abs() <= 1.5
    summary['interval_fu_check'] = dict(
        matches_own_imaging=int(c.interval_fu_matches_imaging.sum()),
        matches_next_patient_imaging=c[c.interval_fu_matches_next_patient_imaging & ~c.interval_fu_matches_imaging].TCIA_ID.tolist(),
        missing=c[c.Interval_FU.isna()].TCIA_ID.tolist(), negative=c[c.Interval_FU < 0].TCIA_ID.tolist(),
        other_mismatch=c[c.Interval_FU.notna() & ~c.interval_fu_matches_imaging & ~c.interval_fu_matches_next_patient_imaging].TCIA_ID.tolist())

    # ---------------- transition manifest
    selp = sel.set_index(['patient_id', 'timepoint'])
    rows = []
    for i, r in c.iterrows():
        p = r.TCIA_ID
        t = tr.loc[i]
        dt = (tp.loc[p, 'fu_date'] - tp.loc[p, 'bl_date']).days
        flags = []
        if r.Interval_FU < 0:
            flags.append('interval_fu_negative_in_source')
        elif pd.isna(r.Interval_FU):
            flags.append('interval_fu_missing_in_source')
        elif not r.interval_fu_matches_imaging:
            flags.append('interval_fu_row_shift_suspected' if r.interval_fu_matches_next_patient_imaging else 'interval_fu_mismatch_imaging')
        if i >= c.index[c.TCIA_ID == 'HCC_082'][0]:
            flags.append('ttp_os_row_alignment_unverifiable')  # Interval_FU is shifted from HCC_082 on; measurements verified aligned
        if pd.isna(r.Tr_Size):
            flags.append('tr_size_missing')
        for tag in (selp.loc[(p, 'BL'), 'quality_flags'], selp.loc[(p, 'FU'), 'quality_flags']):
            flags += [x for x in str(tag).split(';') if x and x != 'nan']
        if 'Cisplastin' in str(r.chemotherapy):
            flags.append('treatment_spelling_normalized')
        for crit in ('RECIST', 'mRECIST'):
            if r[f'{crit}_n_readers'] > 0 and not r[f'{crit}_all_agree']:
                flags.append(f'{crit.lower()}_reader_disagreement')
            if r[f'{crit}_n_readers'] and r[f'{crit}_n_readers'] < 3:
                flags.append(f'{crit.lower()}_fewer_than_3_readers')
        excl = []
        if tp.loc[p, 'same_day'] or dt <= 0:
            excl.append('follow-up not after baseline')
        if t.treatment_normalized == 'UNKNOWN':
            excl.append('treatment unknown')
        if pd.isna(r.mRECIST_BL_median) or pd.isna(r.mRECIST_FU_median):
            excl.append('no follow-up measurement')
        rows.append(dict(
            patient_id=p,
            baseline_study_uid=tp.loc[p, 'bl_study'], baseline_study_date=tp.loc[p, 'bl_date'].date().isoformat(),
            baseline_ct_series_uid=selp.loc[(p, 'BL'), 'series_uid'], baseline_seg_uid_if_available=segp.loc[p, 'seg_series_uid'],
            treatment_date_if_available='', treatment_raw=r.chemotherapy, treatment_normalized=t.treatment_normalized,
            treatment_agents=t.agents, treatment_embolic_material=t.embolic_material,
            followup_study_uid=tp.loc[p, 'fu_study'], followup_study_date=tp.loc[p, 'fu_date'].date().isoformat(),
            followup_ct_series_uid=selp.loc[(p, 'FU'), 'series_uid'], followup_seg_uid_if_available='',
            delta_t_days=dt, delta_t_weeks=round(dt / 7, 2), delta_t_source='DICOM StudyDate (BL study -> FU study)',
            interval_fu_weeks_source=r.Interval_FU, interval_diagnosis_to_bl_days_source=r.Interval_BL,
            **{f'baseline_{k.lower()}_measurement': r[f'{k}_BL_median'] for k in ('RECIST', 'mRECIST')},
            **{f'followup_{k.lower()}_measurement': r[f'{k}_FU_median'] for k in ('RECIST', 'mRECIST')},
            baseline_easl_measurement=r['1_EASL_BL'], followup_easl_measurement=r['1_EASL_FU'],
            **{f'r{rd}_{k.lower()}_{s_.lower()}': r[f'{rd}_{k}_{s_}'] for k in ('RECIST', 'mRECIST') for rd in READERS for s_ in ('BL', 'FU')},
            recist_response_raw=json.dumps([None if pd.isna(r[f'{rd}_RECIST']) else int(r[f'{rd}_RECIST']) for rd in READERS]),
            mrecist_response_raw=json.dumps([None if pd.isna(r[f'{rd}_mRECIST']) else int(r[f'{rd}_mRECIST']) for rd in READERS]),
            easl_response_raw=None if pd.isna(r['1_EASL']) else int(r['1_EASL']),
            recist_majority=r.RECIST_majority, recist_median_rule=r.RECIST_median_derived,
            mrecist_majority=r.mRECIST_majority, mrecist_median_rule=r.mRECIST_median_derived,
            ttp=r.TTP, overall_survival=r.OS, progressed=r.Censored_0_progressed_1, died=r.Death_1_StillAliveorLostToFU_0,
            baseline_clinical_available=bool(r[BASELINE_CLINICAL].drop(['Tr_Size']).notna().all()),
            treatment_available=t.treatment_normalized != 'UNKNOWN',
            followup_measurement_available=pd.notna(r.mRECIST_FU_median),
            transition_eligible=not excl, exclusion_reason=';'.join(excl), data_quality_flags=';'.join(dict.fromkeys(flags))))
    tm = pd.DataFrame(rows)
    tm.to_csv(out / 'transition_manifest.csv', index=False)

    # ---------------- cohort flow
    flow = [('patients (clinical rows = imaging patients)', len(tm)),
            ('BL and FU CT studies on different dates', int((tm.delta_t_days > 0).sum())),
            ('+ follow-up measurement (mRECIST/RECIST)', int(((tm.delta_t_days > 0) & tm.followup_measurement_available).sum())),
            ('+ treatment known = PRIMARY transition cohort', int(tm.transition_eligible.sum())),
            ('state-only cohort (measurement, treatment unknown)', int((tm.followup_measurement_available & ~tm.treatment_available).sum()))]
    pd.DataFrame(flow, columns=['step', 'n']).to_csv(out / 'cohort_flow.csv', index=False)
    summary['cohort_flow'] = dict(flow)
    summary['exclusions'] = tm.exclusion_reason.replace('', 'eligible').value_counts().to_dict()

    # ---------------- treatment-group overlap (diagnostic only)
    e = tm[tm.transition_eligible].merge(c, left_on='patient_id', right_on='TCIA_ID').merge(seg[['patient_id', 'mass_volume_ml', 'mass_components_3d', 'mass_to_liver_ratio']], on='patient_id')
    e['deb'] = (e.treatment_normalized == 'DEB-TACE').astype(int)
    feats = pd.DataFrame({
        'age': e.age, 'female': (e.Sex == 2).astype(int), 'hbv_or_hcv': (e.hepatitis != 'No virus').astype(int),
        'child_pugh_B_or_C': e.CPS.isin(['B', 'C']).astype(int), 'bclc_C_or_D': e.BCLC.isin(['Stage-C', 'Stage-D']).astype(int),
        'multinodular': (e.tumor_nodul == 'multinodular').astype(int), 'involvement_gt50': (e.T_involvment == '>50%').astype(int),
        'log_afp': np.log10(e.AFP.clip(lower=1)), 'portal_vein_thrombosis': e['Portal Vein Thrombosis'], 'vascular_invasion': e['Vascular invasion'],
        'metastasis': e.Metastasis, 'ps_bclc': e['PS_bclc_0_0_1-2_1_3-4_3'], 'baseline_mrecist_mm': e.baseline_mrecist_measurement,
        'baseline_recist_mm': e.baseline_recist_measurement, 'log_seg_mass_ml': np.log10(e.mass_volume_ml.clip(lower=1e-3)),
        'mass_to_liver_ratio': e.mass_to_liver_ratio, 'seg_mass_components': e.mass_components_3d, 'delta_t_weeks': e.delta_t_weeks})
    po = []
    for k in feats:
        a_, b_ = feats[k][e.deb == 0], feats[k][e.deb == 1]
        po.append(dict(variable=k, cTACE_mean=a_.mean(), cTACE_sd=a_.std(), DEB_mean=b_.mean(), DEB_sd=b_.std(), SMD_DEB_minus_cTACE=smd(b_, a_),
                       note='follow-up horizon, NOT a baseline covariate; excluded from propensity model' if k == 'delta_t_weeks' else ''))
    X = feats.drop(columns=['delta_t_weeks']).fillna(feats.median(numeric_only=True))
    ps = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)), X, e.deb,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    lo_, hi_ = max(ps[e.deb == 1].min(), ps[e.deb == 0].min()), min(ps[e.deb == 1].max(), ps[e.deb == 0].max())
    ov = dict(n_cTACE=int((e.deb == 0).sum()), n_DEB=int(e.deb.sum()), cv_auc=float(roc_auc_score(e.deb, ps)),
              common_support=[float(lo_), float(hi_)], frac_in_common_support=float(((ps >= lo_) & (ps <= hi_)).mean()),
              frac_ps_between_0p1_0p9=float(((ps > .1) & (ps < .9)).mean()),
              n_abs_smd_gt_0p1=int(sum(abs(x['SMD_DEB_minus_cTACE']) > .1 for x in po[:-1])),
              n_abs_smd_gt_0p25=int(sum(abs(x['SMD_DEB_minus_cTACE']) > .25 for x in po[:-1])))
    pd.DataFrame(po).to_csv(out / 'propensity_overlap.csv', index=False)
    pd.DataFrame({'patient_id': e.patient_id, 'treatment': e.treatment_normalized, 'propensity_DEB_cv': ps}).to_csv(out / 'propensity_scores.csv', index=False)
    summary['propensity'] = ov

    # ---------------- endpoint class counts (primary cohort, overall and by arm)
    ec = {}
    for k in ('mrecist_median_rule', 'mrecist_majority', 'recist_median_rule', 'recist_majority'):
        ec[k] = {arm: e[e.treatment_normalized == arm][k].fillna('no majority').value_counts().to_dict() for arm in ('cTACE', 'DEB-TACE')}
        ec[k]['all'] = e[k].fillna('no majority').value_counts().to_dict()
    ch = (e.followup_mrecist_measurement - e.baseline_mrecist_measurement) / e.baseline_mrecist_measurement
    ec['mrecist_direction(+-20%)'] = pd.cut(ch, [-np.inf, -.2, .2, np.inf], labels=['decrease', 'stable', 'increase']).value_counts().to_dict()
    ec['mrecist_objective_response(CR+PR)'] = e.mrecist_median_rule.isin(['CR', 'PR']).value_counts().to_dict()
    ec['progressed_event'] = e.progressed.value_counts().to_dict()
    ec['died_event'] = e.died.value_counts().to_dict()
    summary['endpoint_class_counts'] = ec
    summary['delta_t_weeks'] = {arm: e[e.treatment_normalized == arm].delta_t_weeks.describe().round(2).to_dict() for arm in ('cTACE', 'DEB-TACE')}
    summary['delta_t_weeks']['all'] = e.delta_t_weeks.describe().round(2).to_dict()
    summary['future_fields_never_in_S_t'] = FUTURE_FIELDS
    summary['baseline_clinical_fields'] = BASELINE_CLINICAL
    summary['missingness_primary_cohort'] = {k: int(v) for k, v in e[BASELINE_CLINICAL].isna().sum().items() if v}
    (out / 'audit_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ('cohort_flow', 'exclusions', 'propensity')}, indent=1, default=str))


def sel_row(p, tp_, r, uid, why, seg_ref, flags):
    return dict(patient_id=p, timepoint=tp_, study_uid=r.StudyInstanceUID, study_date=r.StudyDate, series_uid=uid,
                series_description=r.SeriesDescription, image_type=r.ImageType, phase_if_known=phase_from_metadata(r.SeriesDescription),
                slice_thickness=r.SliceThickness, n_images=r.n_files, z_extent_mm=r.z_extent_mm, selection_reason=why,
                seg_reference=seg_ref, quality_flags=flags)


def resolve_seg_by_geometry(s, patient, seg_uid):
    import pydicom
    segrow = s[s.SeriesInstanceUID == seg_uid].iloc[0]
    ds = pydicom.dcmread(next(Path(segrow.series_dir).glob('*.dcm')), stop_before_pixels=True)
    zs = {round(float(f.PlanePositionSequence[0].ImagePositionPatient[2]), 2) for f in ds.PerFrameFunctionalGroupsSequence}
    hits = []
    for _, r in s[(s.PatientID == patient) & (s.Modality == 'CT')].iterrows():
        cz = {round(float(pydicom.dcmread(f, stop_before_pixels=True, specific_tags=['ImagePositionPatient']).ImagePositionPatient[2]), 2)
              for f in Path(r.series_dir).glob('*.dcm')}
        if zs <= cz:
            hits.append(r.SeriesInstanceUID)
    assert len(hits) == 1, f'{patient}: SEG slice positions match {len(hits)} CT series'
    return hits[0]


if __name__ == '__main__':
    main()
