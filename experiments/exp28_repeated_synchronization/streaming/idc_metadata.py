"""Metadata-only access to the audited IDC series index."""
from ingestion.metadata_loader import derived_kind, load_series


def ispy2_series():
    data, metadata = load_series('ispy2')
    kinds = data.SeriesDescription.map(derived_kind)
    return data.assign(kind=[value[0] for value in kinds], laterality_kind=[value[1] for value in kinds]), metadata
