#!/usr/bin/env python3
"""LLM systems of Exp 3.1 on the shared MU-Glioma-Post timeline (same harness, tools, model and visit clock as the multi-visit runs).
  longcontext : stateless episode + the complete records of every available visit in the prompt (A: natural, late; B: revision)
  stateless / stateful : unchanged Exp 1 agents, revision timeline only (B)
Raw format identical to experiments/multivisit_longitudinal/run_agents.py (resumable; sequences with API errors are re-run).
  OPENAI_BASE_URL=http://127.0.0.1:8610/v1 OPENAI_API_KEY=dummy python run_llm.py --system longcontext --mode natural --repeats 3"""
import argparse
import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import MV, OUT, REV_NOTE, A, Q, T, cohort, install_revision, steps
import agent_adapter as AA  # noqa: E402  (read-only)

LC_NOTE = ('Complete records of every visit of this patient that is available now (chronological order). You may use them instead '
           'of calling tools.')


class RevQueryMap(AA.QueryMap):
    def __getitem__(self, qid):
        text, fmt = super().__getitem__(qid)
        parts = qid.split('|')
        if len(parts) > 2 and parts[2].startswith('rev:'):
            text = REV_NOTE.format(v=parts[2][4:]) + ' ' + text
        return text, fmt


def api_failed(ep):
    return (not ep.get('success', True)) and (ep.get('failure_reason') or '').startswith('RuntimeError: api')


def sequence(client, system, mode, pid, visits, rep):
    G, GSA = AGENTS
    mem = GSA.new_memory() if system == 'stateful' else None
    out, uid = [], uuid.uuid4().hex
    for i, (cur, avail, tag) in enumerate(steps(visits, mode)):
        for q in T.queries_for(pid, avail, cur):
            log = []
            A.VisitClock.set(pid, visits, avail, cur, log)
            qid = T.qid(cur, q, tag)
            t0 = time.time()
            if system == 'stateless':
                ep = G.episode(client, pid, rep, qid)
            elif system == 'longcontext':
                recs = [A.visit_record(pid, v, visits)[0] for v in avail]
                log.clear()
                ep = G.episode(client, pid, rep, qid, note=LC_NOTE + '\n' + json.dumps(recs))
            else:
                ep, res = GSA.episode(client, pid, rep, qid, mem)
                mem = GSA.update_memory(mem, qid, ep, res)
                ep['memory_bytes_after'] = len(json.dumps(mem))
                ep['memory_sha256_after'] = hashlib.sha256(json.dumps(mem, sort_keys=True).encode()).hexdigest()
            ep.update(system=system, mode=mode, step_index=i, current_visit=cur, available=avail, tag=tag, visit_depth=len(avail), query=q,
                      clock_log=[dict(visit=x['visit'], allowed=x['allowed']) for x in log], wall_sec=round(time.time() - t0, 2))
            ep.pop('memory_after', None)
            out.append(ep)
    for ep in out:
        ep['sequence_uid'] = uid
    out.append(dict(system=system, mode=mode, patient_id=pid, repeat_id=rep, query='_end', sequence_uid=uid, api_failed=any(api_failed(e) for e in out)))
    return out


def main():
    global AGENTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, choices=['longcontext', 'stateless', 'stateful'])
    ap.add_argument('--mode', required=True, choices=['natural', 'late', 'revision'])
    ap.add_argument('--repeats', type=int, default=1); ap.add_argument('--workers', type=int, default=32); ap.add_argument('--pilot', type=int, default=0)
    a = ap.parse_args()
    AGENTS = AA.install(MV.parents[1] / 'outputs/multivisit_longitudinal/measurements_frozen.json')
    G = AGENTS[0]
    if a.mode == 'revision':
        install_revision(); G.QUERIES = RevQueryMap()
    coh = cohort(4 if a.mode in ('late', 'revision') else 0)
    if a.pilot:
        coh = dict(list(coh.items())[:a.pilot])
    raw = OUT / ('pilot' if a.pilot else '') / 'raw' / f'{a.system}_{a.mode}.jsonl'
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw.exists():
        for l in raw.read_text().splitlines():
            j = json.loads(l)
            if j['query'] == '_end':
                done[(j['patient_id'], j['repeat_id'])] = not j['api_failed']
    jobs = [(pid, v, r) for r in range(a.repeats) for pid, v in coh.items() if not done.get((pid, r), False)]
    print(f'{a.system} {a.mode}: sequences {len(jobs)} model {G.MODEL}', flush=True)
    from openai import OpenAI
    client = OpenAI()
    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(raw, 'a') as f:
        futs = [ex.submit(sequence, client, a.system, a.mode, pid, v, r) for pid, v, r in jobs]
        for fu in as_completed(futs):
            for ep in fu.result():
                f.write(json.dumps(ep, default=str) + '\n')
            f.flush(); n += 1
            if n % 10 == 0:
                print(n, '/', len(jobs), time.strftime('%H:%M:%S'), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
