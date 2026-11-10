#!/usr/bin/env python3
"""Stateless condition of the correction-retention test (exp22b_retention.py): for each of the same 60 correction events, the
correction is given in the Q1 episode only; Q2 and Q3 are new episodes without memory. Uses exp22b_agent.episode.
  python stateless_correction.py [--workers W]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exp22b_agent as AG  # noqa: E402
sys.path.insert(0, str(HERE))
import local_model_patch  # noqa: E402
local_model_patch.apply()                      # same lenient JSON parser for both agents

OUT = HERE.parents[2] / 'outputs/stateful_baseline'


def main():
    from openai import OpenAI
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=12); a = ap.parse_args()
    evs = json.loads((OUT / 'correction_events.json').read_text())
    raw = OUT / 'stateless_correction_raw.jsonl'
    done = set()
    if raw.exists():
        for l in raw.read_text().splitlines():
            j = json.loads(l)
            if not ((not j['success']) and (j.get('failure_reason') or '').startswith('RuntimeError: api')):
                done.add((j['run_tag'], j['query_id']))
    tag = lambda e: json.dumps({'id': e['correction_id'], 'pb': e['pb'], 'pf': e['pf']})  # noqa: E731
    jobs = [(e, q) for e in evs for q in ('Q1', 'Q2', 'Q3') if (tag(e), q) not in done]
    print('episodes', len(jobs), 'model', AG.MODEL, flush=True)
    client = OpenAI()
    with ThreadPoolExecutor(a.workers) as ex, open(raw, 'a') as f:
        futs = {ex.submit(AG.episode, client, e['patient_id'], 0, q, e['statement'] if q == 'Q1' else None): (e, q) for e, q in jobs}
        for fu in as_completed(futs):
            e, q = futs[fu]; r = fu.result(); r['run_tag'] = tag(e)
            f.write(json.dumps(r) + '\n'); f.flush()
    print('finished')


if __name__ == '__main__':
    main()
