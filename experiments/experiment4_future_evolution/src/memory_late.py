#!/usr/bin/env python3
"""4D delayed arrival, generic-memory arm: the unchanged longitudinal-forecasting memory agent (same prompt, model, settings) receives the T1 record
only after T2. Patients with observed T1 and T2 (636). Resumable. Writes results/raw/memory_ispy2_late.jsonl.
  python memory_late.py [--pilot N] [--workers 64]"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import E29, E4, ISPY2, T2
sys.path.insert(0, str(E29))
import memory_agent as MA  # noqa: E402
from data import vrec  # noqa: E402
from check import check  # noqa: E402

RAW = E4 / 'results/raw/memory_ispy2_late.jsonl'


def late_events(I, p):
    c = I.clinical(p); c.pop('arm')
    ev = [[dict(type='clinical_record', content=c), dict(type='visit_record', visit='T0', record=vrec(I.record(p, 'T0', 'natural', 0)))]
          if I.record(p, 'T0', 'natural', 0) is not None else [dict(type='clinical_record', content=c), dict(type='missing_visit', visit='T0',
                                                                                                       note='no MRI was acquired at this scheduled visit')]]
    ev.append([dict(type='pending_visit', visit='T1', note='the T1 MRI was acquired but its record has not been received yet')])
    ev.append([dict(type='visit_record', visit='T2', record=vrec(I.record(p, 'T2', 'natural', 2))),
               dict(type='late_record', visit='T1', note='delayed record of the T1 MRI (acquired before T2), received only now',
                    record=vrec(I.record(p, 'T1', 'natural', 1)))])
    o3 = I.record(p, 'T3', 'natural', 3)
    ev.append([dict(type='visit_record', visit='T3', record=vrec(o3)) if o3 is not None else
               dict(type='missing_visit', visit='T3', note='no MRI was acquired at this scheduled visit')])
    return ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pilot', type=int, default=0); ap.add_argument('--workers', type=int, default=64); a = ap.parse_args()
    check()
    I = ISPY2()
    pts = [p for p in I.pts if I.record(p, 'T1', 'natural', 1) is not None and I.record(p, 'T2', 'natural', 2) is not None]
    pts = pts[:a.pilot] if a.pilot else pts
    raw = RAW.with_name('pilot_' + RAW.name) if a.pilot else RAW
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['patient_id'] for l in raw.read_text().splitlines()} if raw.exists() else set()
    todo = [p for p in pts if p not in done]
    evs = {p: late_events(I, p) for p in todo}
    client = MA.OpenAI(base_url='http://127.0.0.1:8610/v1', api_key='dummy')
    print(f'late arrival: {len(todo)} patients', flush=True)
    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(raw, 'a') as f:
        futs = {ex.submit(MA.run_patient, client, evs[p]): p for p in todo}
        for fu in as_completed(futs):
            p = futs[fu]; snaps, logs = fu.result()
            f.write(json.dumps(dict(patient_id=p, cohort='ispy2', scenario='late', events=evs[p], snapshots=snaps, logs=logs), default=str) + '\n')
            f.flush(); n += 1
            if n % 100 == 0:
                print(n, '/', len(todo), time.strftime('%H:%M:%S'), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
