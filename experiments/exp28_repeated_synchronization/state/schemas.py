"""Canonical, modality-independent longitudinal patient-state schema (OncoState 2.8)."""

SCHEMA_VERSION = 'oncostate-2.8/1.0'
EXTRACTION_VERSION = 'exp28-extract-1.0'
PROCESSING_VERSION = 'exp28-sync-1.0'

# Required provenance keys on every observed field
PROVENANCE_KEYS = ('source_type', 'source_file', 'source_study_uid', 'source_series_uid', 'extraction_version', 'processing_version')

# Allowed provenance source types; "official" only for values produced by the trial's own pipeline
SOURCE_TYPES = {
    'official_ftv_seg_header': 'UCSF trial FTV result stored in the analysis-mask DICOM SEG header (0117,10B0)',
    'official_trial_measurement_table': 'UCSF multi-feature MRI table distributed by TCIA',
    'official_clinical_spreadsheet': 'TCIA clinical spreadsheet',
    'local_ftv_recomputed': 'FTV recomputed locally from the analysis mask (+ early PE map); NOT the official FTV',
    'synthetic_demo': 'synthetic demonstration value (SYNTHETIC DEMO only)',
}

OBS_STATUS = ('observed', 'not_observed', 'hidden_stress_test', 'late_arrival')

# State sections that hold observations (historical; must never change silently) vs derived (recomputed each version)
HISTORICAL_SECTIONS = ('history',)
DERIVED_SECTIONS = ('tumour_state',)

# Fields that must never be part of the state (outcomes known only after the last synchronized visit)
FORBIDDEN_STATE_FIELDS = ('pCR', 'pcr', 'RCBClass', 'RFS', 'rfs_ind', 'sstat', 'survDtD2 (tx)')
