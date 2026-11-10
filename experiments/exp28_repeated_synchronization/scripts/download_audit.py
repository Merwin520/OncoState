"""Download-size audit. Exact series sizes from the IDC index; writes DOWNLOAD_AUDIT.md and exports/download_audit.csv.

Two quantities matter and are reported separately:
  transfer   = bytes moved over the network in total;
  peak cache = largest amount held locally at once under patient-by-patient streaming (the 10 GB HARD limit applies here).
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import EXPORTS, PKG, config  # noqa: E402

GB = 2**30
TIER2_PER_SUBTYPE = 6


def tier2_subset(vm, pm):
    """Deterministic, outcome-blind imaging-grounded subset: complete T0-T3, standard PE threshold at every visit,
    unilateral cropped objects; per clinical subtype, the first patients by SHA-256 of the patient ID."""
    std = vm.groupby('patient_id').apply(lambda g: (~g.QC_flags.fillna('').str.contains('nonstandard_PE')).all()
                                         and (g.mask_laterality == 'unilateral_cropped').all(), include_groups=False)
    el = pm[pm.complete_T0_T1_T2_T3 & pm.patient_id.astype(str).map(std)].copy()
    el['h'] = el.patient_id.astype(str).map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    return el.sort_values('h').groupby('clinical_subtype').head(TIER2_PER_SUBTYPE).patient_id.astype(str).tolist(), len(el)


def main():
    vm = pd.read_csv(EXPORTS / 'ispy2_visit_manifest.csv', dtype={'patient_id': str})
    pm = pd.read_csv(EXPORTS / 'ispy2_patient_manifest.csv', dtype={'patient_id': str})
    vm['bytes_dce_used'] = vm.bytes_cropped_dce.where(vm.has_cropped_DCE, vm.bytes_derived_dce)
    early_pe = vm.bytes_pe / 2  # ~2 PE maps per study on average; the early map is one of them (upper bound used below)
    strategies = {
        'A clinical + official measurements (SEG header ranged reads, in memory)': ('bytes_header_fetched', 0.0),
        'B FTV analysis masks only (full DICOM SEG)': ('bytes_seg', None),
        'C FTV masks + DCE used for FTV (cropped, or bilateral original)': (['bytes_seg', 'bytes_dce_used'], None),
        'D original DCE used for FTV only': ('bytes_dce_used', None),
        'E all imaging': ('estimated_download_bytes', None)}
    cohorts = {'all 719 patients': pm.patient_id, '>= 3 valid visits': pm[pm.n_valid_visits >= 3].patient_id,
               'complete T0-T3': pm[pm.complete_T0_T1_T2_T3].patient_id}
    t2, n_t2_eligible = tier2_subset(vm, pm)
    rows = []
    for cname, ids in cohorts.items():
        v = vm[vm.patient_id.isin(set(ids))]
        for sname, (cols, fixed_peak) in strategies.items():
            cols = cols if isinstance(cols, list) else [cols]
            per_patient = v.groupby('patient_id')[cols].sum().sum(axis=1)
            rows.append(dict(cohort=cname, strategy=sname, patients=v.patient_id.nunique(), studies=len(v),
                             series=int(len(v) * (0 if sname.startswith('A') else len(cols) if not sname.startswith('E') else v.n_series.mean())),
                             transfer_GB=round(v[cols].sum().sum() / GB, 2),
                             peak_cache_GB_patient_streaming=round(fixed_peak if fixed_peak is not None else per_patient.max() / GB, 3),
                             fits_10GB_cache=bool((fixed_peak if fixed_peak is not None else per_patient.max() / GB) <= config()['max_cache_gb'])))
    v2 = vm[vm.patient_id.isin(t2)]
    t2_cols = ['bytes_seg', 'bytes_cropped_dce']
    t2_transfer = (v2[t2_cols].sum().sum() + early_pe[v2.index].sum()) / GB
    t2_peak = (v2.groupby('patient_id')[t2_cols].sum().sum(axis=1) + early_pe[v2.index].groupby(v2.patient_id).sum()).max() / GB
    rows.append(dict(cohort=f'Tier 2 subset ({len(t2)} patients)', strategy='SEG + early PE map + cropped DCE, streamed per patient',
                     patients=len(t2), studies=len(v2), series=len(v2) * 3, transfer_GB=round(t2_transfer, 2),
                     peak_cache_GB_patient_streaming=round(t2_peak, 3), fits_10GB_cache=bool(t2_peak <= config()['max_cache_gb'])))
    tab = pd.DataFrame(rows)
    tab.to_csv(EXPORTS / 'download_audit.csv', index=False)
    pd.Series(t2, name='patient_id').to_csv(EXPORTS / 'tier2_subset.csv', index=False)
    write_md(tab, t2, n_t2_eligible, vm, pm)
    print(tab.to_string(index=False))


def write_md(tab, t2, n_el, vm, pm):
    full = tab[tab.cohort == 'all 719 patients']
    L = ['# Download audit', '',
         'Sizes are exact per-series sizes from the IDC index (v24), aggregated per study and per patient. '
         'Two quantities are reported: **transfer** (network bytes in total) and **peak cache** (the largest amount held locally at once '
         'when patients are processed one at a time and raw data are deleted after extraction). The 10 GB HARD limit applies to the peak cache.', '',
         '## Strategies × cohorts', '', tab.to_markdown(index=False), '',
         '## Findings', '',
         f"- **Strategy A needs no imaging cache.** The official UCSF FTV for every study sits in private tags of the analysis-mask SEG header "
         f"((0117,10B0) FTV Sequence → (0117,10B4) volume in cc). A 256 KB HTTP range read per SEG, parsed in memory, recovers it: "
         f"{len(vm)} studies, {full.iloc[0].transfer_GB} GB transferred in total (vs {full.iloc[1].transfer_GB} GB for the full SEG objects), "
         f"0 bytes written to disk, 0 failures. Validated against the UCSF multi-feature table (see DATA_AUDIT.md).",
         f"- Full-cohort imaging strategies B–E all exceed 10 GB of *transfer* ({full.iloc[1].transfer_GB}–{full.iloc[4].transfer_GB} GB), so none of them "
         f"may be downloaded and kept. Streamed one patient at a time, each fits under the 10 GB *peak cache* "
         f"(largest single patient: {full.peak_cache_GB_patient_streaming.max():.2f} GB for strategy E).",
         '- **Recommended default:** Tier 1 uses strategy A for all patients (no imaging stored). '
         f"Tier 2 uses a deterministic, outcome-blind subset of {len(t2)} patients ({TIER2_PER_SUBTYPE} per clinical subtype, first by SHA-256 of the "
         f"patient ID, among {n_el} patients with complete T0–T3, standard PE threshold and unilateral cropped objects). "
         'Each study downloads only the analysis-mask SEG, the early PE map selected by the SEG header\'s SER timing indices (0117,1035), '
         'and the cropped DCE (for figure thumbnails). Raw files are deleted after extraction and validation.',
         f"- Tier 2: transfer {tab.iloc[-1].transfer_GB} GB, peak cache {tab.iloc[-1].peak_cache_GB_patient_streaming} GB (one patient at a time).",
         '- The storage guard refuses to start any download whose size (from the IDC index) plus current cache use would exceed the configured cap.',
         '', 'Tier 2 patients: `exports/tier2_subset.csv`.']
    (PKG / 'DOWNLOAD_AUDIT.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
