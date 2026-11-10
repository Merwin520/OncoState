"""Harvest official FTV results from every I-SPY2 analysis-mask SEG header (ranged reads, in memory, resumable).

Output (work dir): metadata/ispy2_seg_headers.jsonl — one line per SEG, including errors. Re-running skips done SEGs.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.metadata_loader import load_series  # noqa: E402
from paths import metadata_dir  # noqa: E402
from streaming import seg_header  # noqa: E402


def main(workers=16):
    series, meta = load_series('ispy2')
    seg = series[series.Modality == 'SEG']
    out = metadata_dir() / 'ispy2_seg_headers.jsonl'
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if not r.get('error'):
                done.add(r['seg_series_uid_index'])
    todo = seg[~seg.SeriesInstanceUID.isin(done)]
    print(f'{len(seg)} SEG, {len(done)} done, {len(todo)} to fetch (IDC {meta.get("idc_version")})')
    t0 = time.time()

    def job(r):
        try:
            return dict(seg_header.fetch(r.series_aws_url), seg_series_uid_index=r.SeriesInstanceUID, study_uid_index=r.StudyInstanceUID,
                        patient_index=r.PatientID, error='')
        except Exception as e:  # recorded, never dropped silently
            return dict(seg_series_uid_index=r.SeriesInstanceUID, study_uid_index=r.StudyInstanceUID, patient_index=r.PatientID, error=repr(e)[:300])

    with ThreadPoolExecutor(workers) as ex, out.open('a') as fh:
        for i, fut in enumerate(as_completed([ex.submit(job, r) for r in todo.itertuples()])):
            fh.write(json.dumps(fut.result(), default=str) + '\n')
            if i % 250 == 0:
                fh.flush()
                print(i, round(time.time() - t0, 1), 's', flush=True)
    print('done', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
