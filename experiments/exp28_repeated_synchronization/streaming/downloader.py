"""Exact-series downloader from the official IDC public bucket (HTTPS, no credentials).

The series' object prefix comes from the IDC index (`series_aws_url`). Objects are listed, fetched,
size-checked against the listing, and written to the visit cache directory. This avoids holding the multi-GB IDC
index in the processing process (the earlier idc-index-based downloader held it in memory). Objects are fetched in
parallel threads, written atomically (.part -> final name), and retried on transient network errors.
"""
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
from pathlib import Path


def _get(url, timeout=120, attempts=4):
    """GET with retries on transient network errors (timeouts, TLS handshake failures)."""
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if k == attempts - 1:
                raise
            time.sleep(2 ** k)


def list_objects(series_aws_url):
    bucket, prefix = series_aws_url.split('/')[2], series_aws_url.split('/')[3]
    keys, sizes, token = [], [], None
    while True:
        url = f'https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}/' + (f'&continuation-token={urllib.parse.quote(token)}' if token else '')
        x = _get(url).decode()
        keys += re.findall(r'<Key>([^<]+)</Key>', x)
        sizes += [int(s) for s in re.findall(r'<Size>(\d+)</Size>', x)]
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', x)
        if not m:
            return bucket, list(zip(keys, sizes))
        token = m.group(1)


class SeriesDownloader:
    def __init__(self, series_table, workers=16):
        self.workers = workers
        self.url = dict(zip(series_table.SeriesInstanceUID, series_table.series_aws_url))
        self.bytes_downloaded = 0

    def download(self, series_uids, destination):
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        for uid in series_uids:
            bucket, objects = list_objects(self.url[uid])
            todo = []
            for key, size in objects:
                f = dest / f"{uid}__{key.split('/')[-1]}"
                if not (f.exists() and f.stat().st_size == size):   # resume: an intact object is not fetched again
                    todo.append((key, size, f))

            def fetch(item):
                key, size, f = item
                b = _get(f'https://{bucket}.s3.amazonaws.com/{key}')
                if len(b) != size:
                    raise IOError(f'corrupted download {key}: {len(b)} != {size} bytes')
                tmp = f.with_suffix('.part')
                tmp.write_bytes(b)
                tmp.replace(f)              # a file appears only when complete
                return size

            with ThreadPoolExecutor(self.workers) as ex:
                self.bytes_downloaded += sum(ex.map(fetch, todo))
