"""I-SPY2 preparation entry point.

python scripts/prepare_ispy2.py --metadata-only                      # IDC index + clinical + SEG-header FTV + manifests
python scripts/prepare_ispy2.py --process-eligible --max-cache-gb 10   # imaging-grounded subset (streamed; Tier 2)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--metadata-only', action='store_true')
    g.add_argument('--process-eligible', action='store_true')
    ap.add_argument('--max-cache-gb', type=float, default=10)
    a = ap.parse_args()
    if a.metadata_only:
        from ingestion.cohort_builder import build
        from ingestion.metadata_loader import load_series
        from scripts.harvest_seg_headers import main as harvest
        load_series('ispy2'); load_series('ispy1')
        harvest()
        vm, pm, flow, *_ = build()
        for k, n in flow:
            print(f'{n:5d}  {k}')
    else:
        from streaming.processor import process_eligible
        process_eligible(max_cache_gb=a.max_cache_gb)


if __name__ == '__main__':
    main()
