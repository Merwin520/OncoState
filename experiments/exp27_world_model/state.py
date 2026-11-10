"""Experiment 2.7: baseline patient state S_BL, action A_TACE, horizon dt_BL->FU and observed future state S_FU.

S_BL is built as a patient-state document (docs/DATA_MODEL.md style: typed sections + per-field provenance) from
pre-treatment information only, then flattened to a low-dimensional feature vector. No radiomics, no phase-dependent
features. Every feature names its source; build() fails loudly if any source is a future/outcome field.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import audit_transitions as A

REPO = Path(__file__).resolve().parents[2]
AUDIT_RUN = REPO / 'outputs/exp27/audit/run'
STATE_SCHEMA_VERSION = 'exp27-state-1.0'

# feature -> (section, source description, source columns used). Order is the model column order.
S_FEATURES = {
    'bl_mrecist_mm': ('disease', 'median of readers 1-3 baseline mRECIST sum (mm)', ['1_mRECIST_BL', '2_mRECIST_BL', '3_mRECIST_BL']),
    'bl_recist_mm': ('disease', 'median of readers 1-3 baseline RECIST 1.1 sum (mm)', ['1_RECIST_BL', '2_RECIST_BL', '3_RECIST_BL']),
    'log_bl_easl': ('disease', 'log10(1 + reader-1 baseline EASL product)', ['1_EASL_BL']),
    'log_seg_mass_ml': ('imaging', 'log10 of SEG "Mass" volume (mL), baseline SEG', ['SEG:Mass']),
    'log1p_seg_mass_components': ('imaging', 'log(1 + number of 3D connected components of SEG "Mass")', ['SEG:Mass']),
    'seg_largest_component_frac': ('imaging', 'largest Mass component volume / total Mass volume (burden concentration)', ['SEG:Mass']),
    'log_seg_liver_ml': ('imaging', 'log10 of SEG "Liver" volume (mL)', ['SEG:Liver']),
    'seg_mass_to_liver': ('imaging', 'SEG Mass volume / SEG Liver volume', ['SEG:Mass', 'SEG:Liver']),
    'age': ('clinical', 'age at diagnosis (years)', ['age']),
    'female': ('clinical', 'Sex == 2', ['Sex']),
    'viral_hepatitis': ('clinical', 'hepatitis != "No virus"', ['hepatitis']),
    'cirrhosis': ('clinical', 'Evidence_of_cirh', ['Evidence_of_cirh']),
    'child_pugh_b_or_c': ('clinical', 'CPS in {B, C}', ['CPS']),
    'ps_bclc': ('clinical', 'performance status for BCLC (0 / 1 = 1-2 / 3 = 3-4)', ['PS_bclc_0_0_1-2_1_3-4_3']),
    'bclc_stage': ('clinical', 'BCLC stage A..D -> 1..4', ['BCLC']),
    'clip_score': ('clinical', 'CLIP score 0-6', ['CLIP_Score']),
    'multinodular': ('clinical', 'tumor_nodul == multinodular', ['tumor_nodul']),
    'involvement_gt50': ('clinical', 'T_involvment == >50%', ['T_involvment']),
    'vascular_invasion': ('clinical', 'Vascular invasion', ['Vascular invasion']),
    'portal_vein_thrombosis': ('clinical', 'Portal Vein Thrombosis', ['Portal Vein Thrombosis']),
    'metastasis': ('clinical', 'Metastasis', ['Metastasis']),
    'lymph_nodes': ('clinical', 'Lymphnodes', ['Lymphnodes']),
    'log_afp': ('clinical', 'log10(AFP ng/mL, floored at 1)', ['AFP']),
}
ACTION_FEATURE = 'deb_tace'   # A_TACE: 1 = DEB-TACE, 0 = cTACE
HORIZON_FEATURE = 'dt_weeks'  # BL CT -> FU CT (DICOM StudyDate); NOT treatment-to-follow-up time


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def check_no_leakage():
    """Fail loudly if any state feature is sourced from a future / outcome column."""
    bad = {f: src for f, (_, _, src) in S_FEATURES.items()
           if set(src) & set(A.FUTURE_FIELDS) or any(s.endswith('_FU') or s in ('TTP', 'OS') for s in src)}
    if bad:
        raise RuntimeError(f'future information in S_BL: {bad}')
    allowed = set(A.BASELINE_CLINICAL) | {f'{r}_{k}_BL' for r in A.READERS for k in ('RECIST', 'mRECIST')} | {'1_EASL_BL'}
    unknown = {f: src for f, (_, _, src) in S_FEATURES.items() if any(not s.startswith('SEG:') and s not in allowed for s in src)}
    if unknown:
        raise RuntimeError(f'S_BL feature sourced from a column not classified as baseline: {unknown}')
    return {'state_features': list(S_FEATURES), 'future_fields_blocked': A.FUTURE_FIELDS, 'all_sources_baseline': True}


def build(audit_run=AUDIT_RUN):
    """Return (states, frame). states: patient-state documents; frame: one row per primary-cohort patient with
    S_BL features, A, dt, targets (mRECIST % change, FU measurement, ORR; reader-specific variants)."""
    check_no_leakage()
    tm = pd.read_csv(audit_run / 'transition_manifest.csv')
    seg = pd.read_csv(audit_run / 'segmentation_manifest.csv').set_index('patient_id')
    c = pd.read_excel(A.CLINICAL, sheet_name='data table').set_index('TCIA_ID')
    tm = tm[tm.transition_eligible].set_index('patient_id')
    states, rows = [], []
    for p, t in tm.iterrows():
        r, sg = c.loc[p], seg.loc[p]
        v = {
            'bl_mrecist_mm': t.baseline_mrecist_measurement, 'bl_recist_mm': t.baseline_recist_measurement,
            'log_bl_easl': np.log10(1 + r['1_EASL_BL']) if pd.notna(r['1_EASL_BL']) else np.nan,
            'log_seg_mass_ml': np.log10(max(sg.mass_volume_ml, 1e-3)), 'log1p_seg_mass_components': np.log1p(sg.mass_components_3d),
            'seg_largest_component_frac': sg.mass_largest_component_ml / sg.mass_volume_ml if sg.mass_volume_ml > 0 else np.nan,
            'log_seg_liver_ml': np.log10(sg.liver_volume_ml), 'seg_mass_to_liver': sg.mass_to_liver_ratio,
            'age': r.age, 'female': int(r.Sex == 2), 'viral_hepatitis': int(r.hepatitis != 'No virus'), 'cirrhosis': r.Evidence_of_cirh,
            'child_pugh_b_or_c': int(r.CPS in ('B', 'C')), 'ps_bclc': r['PS_bclc_0_0_1-2_1_3-4_3'],
            'bclc_stage': {'Stage-A': 1, 'Stage-B': 2, 'Stage-C': 3, 'Stage-D': 4}.get(r.BCLC, np.nan),
            'clip_score': float(str(r.CLIP_Score).split('_')[-1]) if pd.notna(r.CLIP_Score) else np.nan,
            'multinodular': int(r.tumor_nodul == 'multinodular'), 'involvement_gt50': int(r.T_involvment == '>50%'),
            'vascular_invasion': r['Vascular invasion'], 'portal_vein_thrombosis': r['Portal Vein Thrombosis'], 'metastasis': r.Metastasis,
            'lymph_nodes': r.Lymphnodes, 'log_afp': np.log10(max(r.AFP, 1)) if pd.notna(r.AFP) else np.nan}
        doc = {'patient_id': p, 'timepoint': 'pre_TACE', 'schema_version': STATE_SCHEMA_VERSION,
               'study_uid': t.baseline_study_uid, 'study_date': t.baseline_study_date,
               'imaging': {}, 'clinical': {}, 'disease': {}, 'treatment_history': {'prior_treatment_if_available': None},
               'missingness': {f: bool(pd.isna(x)) for f, x in v.items()}, 'provenance': {}}
        for f, (sec, desc, src) in S_FEATURES.items():
            doc[sec][f] = _num(v[f])
            doc['provenance'][f] = {'derivation': desc, 'source': [s if s.startswith('SEG:') else f'{A.CLINICAL.name}:{s}' for s in src],
                                    'seg_series_uid': sg.seg_series_uid if any(s.startswith('SEG:') for s in src) else None}
        states.append(doc)
        pct = 100 * (t.followup_mrecist_measurement - t.baseline_mrecist_measurement) / t.baseline_mrecist_measurement
        row = dict(patient_id=p, **v, deb_tace=int(t.treatment_normalized == 'DEB-TACE'), dt_weeks=t.delta_t_weeks,
                   y_pct=pct, y_fu_mm=t.followup_mrecist_measurement, orr=int(pct <= -30), consensus_category=t.mrecist_median_rule)
        for rd in A.READERS:
            b, f_ = t[f'r{rd}_mrecist_bl'], t[f'r{rd}_mrecist_fu']
            row[f'y_pct_r{rd}'] = 100 * (f_ - b) / b if pd.notna(b) and pd.notna(f_) and b > 0 else np.nan
        rows.append(row)
    fr = pd.DataFrame(rows).set_index('patient_id')
    # ORR from % change must equal the rule-layer consensus (CR + PR)
    assert (fr.orr == fr.consensus_category.isin(['CR', 'PR']).astype(int)).all(), 'ORR (pct <= -30) disagrees with rule-layer consensus'
    return states, fr


def write_states(states, out):
    out.mkdir(parents=True, exist_ok=True)
    for d in states:
        (out / f"{d['patient_id']}.json").write_text(json.dumps(d, indent=1))
