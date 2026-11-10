#!/usr/bin/env python3
"""Systems A and B (and the pre-specified exploratory OncoState-state + LLM reader) on the shared MU-Glioma-Post timeline.

  stateless : exp22b_agent.episode(client, pid, rep, qid)                       - fresh episode per query (Exp 1 harness, unchanged)
  stateful  : generic_stateful_agent.episode(client, pid, rep, qid, mem) + update_memory(...)   - Exp 1 generic memory, unchanged,
              one memory per (patient, repeat) carried across all visits and queries
  onco_llm  : exp22b_agent.episode(..., note=<latest OncoState state version>)   - EXPLORATORY arm (needs run_oncostate.py output)

Every patient sequence runs visit by visit in one thread; before each episode the VisitClock is set to the visits available at that step,
so tools cannot return future visits. A sequence is written only when complete (resumable; sequences with API errors are re-run).
  EXP22B_MODEL=qwen3.5-9b OPENAI_BASE_URL=http://127.0.0.1:8610/v1 OPENAI_API_KEY=dummy \
  python run_agents.py --system stateless|stateful|onco_llm --mode natural|late --repeats 3 --workers 32 [--pilot N] [--mock]
"""
import argparse
import hashlib
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_adapter as AA  # noqa: E402
import mu_glioma_adapter as A  # noqa: E402
import queries as Q  # noqa: E402
import timeline as T  # noqa: E402

OUT = T.OUT
STATE_NOTE = ('OncoState patient state (latest stored version for this patient; structured, versioned longitudinal state built from the '
              'visit records that have arrived). You may use it instead of calling tools.')


def state_view(pid, mode, step_index):
    """Latest OncoState version after synchronizing this step (read from the stored run1 versions)."""
    recs = ONCO[(pid, mode)]
    ver = next(r['state_version'] for r in recs if r['step_index'] == step_index and r['query'] == '_sync')
    lin = json.loads((OUT / f'oncostate_states/{mode}_run1/{pid}/lineage.json').read_text())
    f = next(v['file'] for v in lin['versions'] if v['state_version'] == ver)
    s = json.loads((OUT / f'oncostate_states/{mode}_run1/{pid}/{f}').read_text())
    return {k: s[k] for k in ('patient_id', 'state_version', 'parent_version', 'current_timepoint', 'history', 'observation_status', 'missingness', 'tumour_state')}


def sequence(client, system, mode, pid, visits, rep):
    G, GSA = AGENTS
    mem = GSA.new_memory() if system == 'stateful' else None
    out, uid = [], uuid.uuid4().hex
    for i, (cur, avail, tag) in enumerate(T.steps(visits, mode)):
        for q in T.queries_for(pid, avail, cur):
            log = []
            A.VisitClock.set(pid, visits, avail, cur, log)
            qid = T.qid(cur, q, tag)
            t0 = time.time()
            if system == 'stateless':
                ep = G.episode(client, pid, rep, qid)
            elif system == 'onco_llm':
                ep = G.episode(client, pid, rep, qid, note=STATE_NOTE + '\n' + json.dumps(state_view(pid, mode, i)))
            else:
                ep, res = GSA.episode(client, pid, rep, qid, mem)
                mem = GSA.update_memory(mem, qid, ep, res)
                ep['memory_bytes_after'] = len(json.dumps(mem))
                ep['memory_sha256_after'] = hashlib.sha256(json.dumps(mem, sort_keys=True).encode()).hexdigest()
                ep['memory_visits_after'] = sorted(k.split(':', 1)[1] for k in mem['previous_tool_outputs'] if k.startswith('get_visit_record:'))
            ep.update(system=system, mode=mode, step_index=i, current_visit=cur, available=avail, tag=tag, visit_depth=len(avail), query=q,
                      clock_log=[dict(visit=x['visit'], allowed=x['allowed']) for x in log], wall_sec=round(time.time() - t0, 2))
            ep.pop('memory_after', None)
            out.append(ep)
        if system == 'stateful':
            out.append(dict(system=system, mode=mode, patient_id=pid, repeat_id=rep, step_index=i, current_visit=cur, query='_memory', memory=mem))
    for ep in out:
        ep['sequence_uid'] = uid
    out.append(dict(system=system, mode=mode, patient_id=pid, repeat_id=rep, query='_end', sequence_uid=uid, api_failed=any(api_failed(e) for e in out)))
    return out


def api_failed(ep):
    return (not ep.get('success', True)) and (ep.get('failure_reason') or '').startswith('RuntimeError: api')


def main():
    global AGENTS, ONCO
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, choices=['stateless', 'stateful', 'onco_llm']); ap.add_argument('--mode', default='natural')
    ap.add_argument('--repeats', type=int, default=3); ap.add_argument('--workers', type=int, default=32); ap.add_argument('--pilot', type=int, default=0)
    ap.add_argument('--mock', action='store_true')
    a = ap.parse_args()
    AGENTS = AA.install(OUT / 'measurements_frozen.json')
    coh = T.cohort()
    if a.mode == 'late':
        coh = {p: v for p, v in coh.items() if len(v) >= 4}
    if a.pilot:
        coh = dict(list(coh.items())[:a.pilot])
    sub = OUT / ('pilot' if (a.pilot or a.mock) else '')
    raw = sub / 'raw' / (f'{"mock_" if a.mock else ""}{a.system}_{a.mode}.jsonl')
    raw.parent.mkdir(parents=True, exist_ok=True)
    if a.system == 'onco_llm':
        ONCO = {}
        for l in (sub / f'raw/oncostate_{a.mode}_run1.jsonl').read_text().splitlines():
            r = json.loads(l); ONCO.setdefault((r['patient_id'], a.mode), []).append(r)
    done = {}
    if raw.exists():
        for l in raw.read_text().splitlines():
            j = json.loads(l)
            if j['query'] == '_end':                         # status of the latest complete sequence
                done[(j['patient_id'], j['repeat_id'])] = not j['api_failed']
    jobs = [(pid, v, r) for r in range(a.repeats) for pid, v in coh.items() if not done.get((pid, r), False)]
    G = AGENTS[0]
    print(f'{a.system} {a.mode}: sequences {len(jobs)} model {"MOCK" if a.mock else G.MODEL} prompt sha256 {G.PROMPT_HASH[:16]}', flush=True)
    if a.mock:
        client = MockClient()
    else:
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


class MockClient:
    """Plumbing test only (never used for results): reads every available visit with the tools and answers with the frozen rules."""
    class _R:
        def __init__(self, output, text=''):
            self.output, self.output_text, self.usage = output, text, None

    class _Call:
        type = 'function_call'

        def __init__(self, name, args, i):
            self.name, self.call_id, self.arguments = name, f'c{i}', json.dumps(args)

    def __init__(self):
        self.responses = self

    def create(self, model, input, tools, max_output_tokens):
        user = next(x['content'] for x in input if isinstance(x, dict) and x.get('role') == 'user')
        pid = user.split('\n')[0].split(': ')[1]
        q = user.split('Query: ')[1].split('\n')[0]
        outs = [json.loads(x['output']) for x in input if isinstance(x, dict) and x.get('type') == 'function_call_output']
        if not outs:
            return self._R([self._Call('list_visits', {'patient_id': pid}, 0)])
        vis = [v['visit'] for v in outs[0]['result']['visits']]
        if len(outs) - 1 < len(vis):
            return self._R([self._Call('get_visit_record', {'patient_id': pid, 'timepoint': vis[len(outs) - 1]}, len(outs))])
        recs = [o['result'] for o in outs[1:]]
        ans = Q.reference_answers([(r['visit'], r['measurements']['burden_ml']) for r in recs],
                                  {r['visit']: r['treatments_reported_since_previous_visit'] for r in recs})
        key = next(k for k, (t, _) in Q.TEXT.items() if q.endswith(t))
        return self._R([], json.dumps(ans.get(key, {})))


AGENTS, ONCO = None, None

if __name__ == '__main__':
    main()
