"""Map source study labels to canonical treatment visits, keeping the source terminology and flagging ambiguity."""
import pandas as pd

from paths import config

ORDER = ['T0', 'T1', 'T2', 'T3']


def map_ispy2_visits(series):
    """series: IDC series table. Returns one row per study with its canonical timepoint and QC flags."""
    cfg = config('ispy2')
    st = series.groupby('StudyInstanceUID').agg(
        PatientID=('PatientID', 'first'), n_patient_ids=('PatientID', 'nunique'), StudyDescription=('StudyDescription', 'first'),
        n_descriptions=('StudyDescription', 'nunique'), StudyDate=('StudyDate', 'first'), n_series=('SeriesInstanceUID', 'size'),
        study_size_MB=('series_size_MB', 'sum')).reset_index()
    st['timepoint'] = st.StudyDescription.map(cfg['timepoint_map'])
    st['patient_id'] = st.PatientID.str.replace(cfg['patient_id_prefix'], '', regex=False)
    flags = []
    dup = st.duplicated(['patient_id', 'timepoint'], keep=False) & st.timepoint.notna()
    for i, r in st.iterrows():
        f = []
        if pd.isna(r.timepoint):
            f.append('ambiguous_timepoint')
        if r.n_patient_ids > 1:
            f.append('conflicting_patient_id')
        if r.n_descriptions > 1:
            f.append('multiple_study_descriptions')
        if dup[i]:
            f.append('duplicate_study_for_timepoint')
        flags.append(';'.join(f))
    st['qc_flags'] = flags
    # time ordering: canonical visit order must agree with study dates within a patient
    st['StudyDate'] = pd.to_datetime(st.StudyDate)
    st['tp_rank'] = st.timepoint.map({t: i for i, t in enumerate(ORDER)})
    for p, g in st.dropna(subset=['tp_rank']).groupby('patient_id'):
        g = g.sort_values('tp_rank')
        if not g.StudyDate.is_monotonic_increasing:
            st.loc[g.index, 'qc_flags'] = st.loc[g.index, 'qc_flags'].map(lambda s: ';'.join(filter(None, [s, 'unexpected_time_ordering'])))
    return st
