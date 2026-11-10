"""Field-level provenance records."""
from state.schemas import EXTRACTION_VERSION, PROCESSING_VERSION, PROVENANCE_KEYS, SOURCE_TYPES


def prov(source_type, source_file, source_study_uid=None, source_series_uid=None, **extra):
    if source_type not in SOURCE_TYPES:
        raise ValueError(f'unknown source_type {source_type}')
    return dict(source_type=source_type, source_file=source_file, source_study_uid=source_study_uid, source_series_uid=source_series_uid,
                extraction_version=EXTRACTION_VERSION, processing_version=PROCESSING_VERSION, **extra)


def field(value, provenance, unit=None, qc_flags=()):
    """One observed value with its provenance. value None means missing (never 0)."""
    return dict(value=value, unit=unit, available=value is not None, provenance=provenance, qc_flags=list(qc_flags))


def complete(p):
    """All required provenance keys present; study/series UIDs may be None only for spreadsheet sources."""
    if not isinstance(p, dict):
        return False
    need = [k for k in PROVENANCE_KEYS if k not in ('source_study_uid', 'source_series_uid')]
    if any(not p.get(k) for k in need):
        return False
    if p['source_type'] in ('official_ftv_seg_header', 'local_ftv_recomputed'):
        return bool(p.get('source_study_uid')) and bool(p.get('source_series_uid'))
    return True
