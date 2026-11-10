#!/usr/bin/env python3
"""Condition M: LLM memory-augmented agent (pre-specified analysis plan). Qwen3.5-9B (local vLLM, temperature 0, seed 0) receives the clinical events
of one patient landmark by landmark and returns the complete updated structured memory after each landmark. Snapshots are stored per
landmark; F is computed from them later. Resumable.  python memory_agent.py --cohort ispy2 --scenario natural [--pilot 3]"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from data import OUT, ISPY1, ISPY2



def parse_json_last(text):
    """Last complete JSON object in the reply (identical to stateful_baseline/local_model_patch.parse_json_last)."""
    dec, last, i = json.JSONDecoder(), None, 0
    while True:
        i = text.find('{', i)
        if i < 0:
            return last
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                last = obj
            i = end
        except ValueError:
            i += 1

MODEL = 'qwen3.5-9b'
SYSTEM = """You are a clinical AI assistant that maintains the persistent longitudinal memory of ONE breast-cancer patient during neoadjuvant
therapy. Scheduled MRI visits are T0 (pre-treatment), T1 (~3 weeks), T2 (end of the first regimen) and T3 (pre-surgery).
You receive the clinical events of each visit as JSON. After the events, return the COMPLETE updated memory as ONE JSON object:
{"patient": {<clinical fields>},
 "visits": [{"visit": "T0", "status": "observed" | "missing", "study_day": <number or null>, "ftv_pe_cc": <number or null>,
             "ftv_ser_cc": <number or null>, "longest_diameter_cm": <number or null>, "sphericity": <number or null>,
             "bpe_5slice_mean": <number or null>}, ...],
 "notes": "<short clinical summary of the trajectory so far>"}
Rules: keep every earlier visit exactly as stored unless an event explicitly revises it; when a revised record arrives, replace that
visit's values with the revised ones; record a visit without MRI as status "missing" with null values; copy all numbers exactly as
given (no rounding); include only fields of the schema that you were given values for (others null). Reply with the JSON only."""


def step(client, memory, events):
    msg = f"Current memory:\n{json.dumps(memory) if memory else '(empty)'}\n\nNew clinical events:\n{json.dumps(events)}\n\nReturn the complete updated memory."
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0, seed=0, max_tokens=3000,
                                               messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': msg}],
                                               extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            text = r.choices[0].message.content or ''
            j = parse_json_last(text)
            if isinstance(j, dict) and isinstance(j.get('visits'), list):
                return j, dict(ok=True, attempt=attempt, tokens=r.usage.total_tokens)
            err = 'unparseable memory'
        except Exception as e:  # noqa: BLE001
            err = f'{type(e).__name__}: {e}'
            time.sleep(2)
    return memory, dict(ok=False, error=err)          # agent failure: memory unchanged (counted)


def run_patient(client, events):
    mem, snaps, logs = None, [], []
    for k, ev in enumerate(events):
        mem, lg = step(client, mem, ev)
        snaps.append(mem); logs.append(lg)
    return snaps, logs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', choices=['ispy2', 'ispy1'], required=True)
    ap.add_argument('--scenario', default='natural'); ap.add_argument('--workers', type=int, default=64); ap.add_argument('--pilot', type=int, default=0)
    a = ap.parse_args()
    C = ISPY2() if a.cohort == 'ispy2' else ISPY1()
    pts = C.pts[:a.pilot] if a.pilot else C.pts
    if a.cohort == 'ispy2' and a.scenario == 'revision':
        pts = [p for p in pts if C.record(p, 'T1', 'natural', 0) is not None]
    raw = OUT / ('pilot' if a.pilot else '') / 'raw' / f'memory_{a.cohort}_{a.scenario}.jsonl'
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['patient_id'] for l in raw.read_text().splitlines()} if raw.exists() else set()
    todo = [p for p in pts if p not in done]
    evs = {p: (C.events(p, a.scenario) if a.cohort == 'ispy2' else C.events(p)) for p in todo}
    client = OpenAI(base_url='http://127.0.0.1:8610/v1', api_key='dummy')
    print(f'{a.cohort} {a.scenario}: {len(todo)} patients', flush=True)
    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(raw, 'a') as f:
        futs = {ex.submit(run_patient, client, evs[p]): p for p in todo}
        for fu in as_completed(futs):
            p = futs[fu]; snaps, logs = fu.result()
            f.write(json.dumps(dict(patient_id=p, cohort=a.cohort, scenario=a.scenario, events=evs[p], snapshots=snaps, logs=logs), default=str) + '\n')
            f.flush(); n += 1
            if n % 100 == 0:
                print(n, '/', len(todo), time.strftime('%H:%M:%S'), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
