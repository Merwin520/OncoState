#!/usr/bin/env python3
"""Generic STATEFUL tool-using agent: the controlled baseline between the stateless agent (exp22b_agent.py) and OncoState.

Identical to exp22b_agent.py in model, system prompt, tools (same Toolbox, same cached registration / matcher / lesion tables on the
setting-A masks), MAX_TURNS, query texts and answer formats, output validation and patient cohort. The ONLY difference: a conventional
persistent memory per patient (and per repeat, so repeats stay independent replications), carried across the query sequence
Q1 -> Q2 -> Q3 -> Q4 and shown to the agent at the start of every query:

    memory = {"history": [...], "previous_measurements": {...}, "previous_tool_outputs": {...},
              "previous_correspondences": {...}, "clinician_corrections": {...}}

It is a plain JSON dictionary appended to after every query. It has NO formal versioned state, transition operator, write-once versions,
parent chaining, hashes, field-level provenance or late-arrival semantics. Memory only ever contains what this agent itself obtained
from its tools or answered earlier (or, in the correction run, clinician corrections given after Q1); no reference / GT information.

Usage:  python generic_stateful_agent.py [--pilot N] [--repeats R] [--workers W] [--corrections FILE] [--mock]
  --corrections  correction-persistence run: after Q1, inject the given clinician corrections into memory (see correction_run.py)
  --mock         plumbing test with a scripted fake model (no API calls; never used for results)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exp22b_agent as G  # noqa: E402  (same tools, prompt, model, validation)
sys.path.insert(0, str(HERE))
import local_model_patch  # noqa: E402
local_model_patch.apply()                      # same lenient JSON parser for both agents

OUT = HERE.parents[2] / 'outputs/stateful_baseline'
QORDER = ['Q1', 'Q2', 'Q3', 'Q4']
MEMORY_NOTE = ('Persistent patient memory from your earlier queries in this session (results of tools you called, your previous answers '
               'and any clinician corrections). It is kept between queries; you may reuse it instead of calling tools again. '
               'Clinician corrections override earlier results.')


def new_memory():
    return {'history': [], 'previous_measurements': {}, 'previous_tool_outputs': {}, 'previous_correspondences': {}, 'clinician_corrections': {}}


def update_memory(mem, qid, ep, tool_results):
    """Append this query's tool outputs and answer (conventional memory: later entries simply overwrite earlier keys)."""
    for name, args, out in tool_results:
        key = name if name in ('get_study_info', 'register_studies', 'registered_baseline_centroids', 'match_lesions') else f"{name}:{args.get('timepoint', '')}"
        if name == 'calculate':
            continue
        mem['previous_tool_outputs'][key] = out
        if name == 'list_lesions':
            mem['previous_measurements'][args.get('timepoint', '')] = out
        if name == 'match_lesions':
            mem['previous_correspondences']['matcher_output'] = out
    if ep['structured_answer'] is not None:
        mem['history'].append({'query': qid, 'answer': ep['structured_answer']})
        if qid == 'Q1':
            mem['previous_correspondences']['agent_answer_Q1'] = ep['structured_answer'].get('matches')
    return mem


class RecordingToolbox(G.Toolbox):
    """Same toolbox; additionally keeps the parsed outputs so they can be written to memory."""
    def __init__(self):
        super().__init__(); self.results = []

    def run(self, name, args):
        out = super().run(name, args)
        j = json.loads(out)
        if j.get('ok'):
            self.results.append((name, args, j['result']))
        return out


def episode(client, pid, rep, qid, mem):
    """exp22b_agent.episode with the memory block appended to the user message (everything else identical)."""
    tb = RecordingToolbox(); q, fmt = G.QUERIES[qid]; t0 = time.time(); tokens = 0; final = ''; fail = None; turns = 0
    usage = G.usage_detail(None)
    user = f'Patient id: {pid}\nQuery: {q}\nReturn exactly this JSON format: {fmt}\n\n{MEMORY_NOTE}\n{json.dumps(mem)}'
    items = [{'role': 'developer', 'content': G.SYSTEM}, {'role': 'user', 'content': user}]
    try:
        while turns < G.MAX_TURNS:
            turns += 1
            r = None
            for attempt in range(4):
                try:
                    r = client.responses.create(model=G.MODEL, input=items, tools=G.TOOLS, max_output_tokens=6000); break
                except Exception as e:
                    err = f'{type(e).__name__}: {str(e)[:100]}'; time.sleep(2 + 3 * attempt)
            if r is None:
                raise RuntimeError('api: ' + err)
            tokens += (r.usage.input_tokens + r.usage.output_tokens) if r.usage else 0
            u = G.usage_detail(r.usage); usage = {k: usage[k] + u[k] for k in usage}
            items += list(r.output)
            calls = [it for it in r.output if getattr(it, 'type', '') == 'function_call']
            if not calls:
                final = r.output_text or ''; break
            for it in calls:
                try:
                    args = json.loads(it.arguments)
                except Exception:
                    args = {}
                items.append({'type': 'function_call_output', 'call_id': it.call_id, 'output': tb.run(it.name, args)})
        else:
            fail = 'max_turns'
    except Exception as e:
        fail = f'{type(e).__name__}: {str(e)[:120]}'
    j = G.parse_json(final) if final else None
    ok = fail is None and j is not None and G.validate(qid, j)
    if fail is None and not ok:
        fail = 'invalid_or_missing_json'
    ep = dict(patient_id=pid, repeat_id=rep, query_id=qid, raw_answer=final, structured_answer=j if ok else None, tool_calls=tb.calls,
              tools_used=sorted({c['name'] for c in tb.calls}), n_tool_calls=len(tb.calls), n_turns=turns, latency_sec=round(time.time() - t0, 2),
              token_count=tokens, usage=usage, success=bool(ok), failure_reason=None if ok else fail,
              seen_baseline_ids=sorted(tb.seen_ids['baseline']), seen_followup_ids=sorted(tb.seen_ids['followup']),
              memory_bytes_in=len(json.dumps(mem)))
    return ep, tb.results


def patient_sequence(client, pid, rep, corrections=None, tag=None):
    """Q1 -> Q4 for one patient and repeat, carrying the memory. Corrections (correction run only) are written into memory before Q1,
    i.e. at the same point at which the existing retention test (exp22b_retention.py) gives them to the stateless agent."""
    mem, out = new_memory(), []
    for c in corrections or []:
        mem['clinician_corrections'][c['correction_id']] = {'baseline_lesion_id': c['pb'], 'followup_lesion_id': c['pf'], 'statement': c['statement']}
    for qid in QORDER:
        ep, res = episode(client, pid, rep, qid, mem)
        mem = update_memory(mem, qid, ep, res)
        ep['memory_after'] = json.loads(json.dumps(mem)); ep['run_tag'] = tag
        out.append(ep)
    return out


class MockClient:
    """Scripted stand-in for plumbing tests only: calls the tools a careful agent would call (skipping those already in memory) and
    answers from them. Never used for any reported result."""
    class _R:
        def __init__(self, output, text=''):
            self.output, self.output_text, self.usage = output, text, None

    class _Call:
        type = 'function_call'
        def __init__(self, name, pid, i, tp=None):
            self.name, self.call_id = name, f'c{i}'
            self.arguments = json.dumps({'patient_id': pid, **({'timepoint': tp} if tp else {})})

    def __init__(self):
        self.responses = self

    def create(self, model, input, tools, max_output_tokens):
        user = next(x['content'] for x in input if isinstance(x, dict) and x.get('role') == 'user')
        pid = user.split('\n')[0].split(': ')[1]; mem = json.loads(user.split(MEMORY_NOTE + '\n')[1])
        outs = [json.loads(x['output']) for x in input if isinstance(x, dict) and x.get('type') == 'function_call_output']
        need = [n for n in ('register_studies', 'match_lesions') if n not in mem['previous_tool_outputs']]
        need += [f'list_lesions:{tp}' for tp in ('baseline', 'followup') if f'list_lesions:{tp}' not in mem['previous_tool_outputs']]
        if len(outs) < len(need):
            n = need[len(outs)]; nm, _, tp = n.partition(':')
            return self._R([self._Call(nm, pid, len(outs), tp or None)])
        res = {**mem['previous_tool_outputs'], **{n: o['result'] for n, o in zip(need, outs)}}
        m = res['match_lesions']
        pairs = [{'baseline_lesion_id': x['baseline_lesion_id'], 'followup_lesion_id': x['followup_lesion_id']} for x in m['matches']]
        bl = {x['id']: x for x in res['list_lesions:baseline']}; fu = {x['id']: x for x in res['list_lesions:followup']}
        for c in mem['clinician_corrections'].values():                      # corrections override earlier pairs
            pairs = [p for p in pairs if p['baseline_lesion_id'] != c['baseline_lesion_id'] and p['followup_lesion_id'] != c['followup_lesion_id']]
            pairs.append({'baseline_lesion_id': c['baseline_lesion_id'], 'followup_lesion_id': c['followup_lesion_id']})
        mb, mf = {p['baseline_lesion_id'] for p in pairs}, {p['followup_lesion_id'] for p in pairs}
        q = user.split('Query: ')[1].split('\n')[0]
        if q == G.QUERIES['Q1'][0]:
            ans = {'matches': pairs}
        elif q == G.QUERIES['Q2'][0]:
            ans = {'baseline_status': {str(i): 'persistent' if i in mb else 'disappearing' for i in bl},
                   'followup_status': {str(i): 'persistent' if i in mf else 'newly_appearing' for i in fu}}
        elif q == G.QUERIES['Q3'][0]:
            ch = lambda a, b: (fu[b]['volume_mm3'] - bl[a]['volume_mm3']) / bl[a]['volume_mm3'] * 100  # noqa: E731
            ans = {'changes': [{**p, 'change': 'enlarged' if ch(p['baseline_lesion_id'], p['followup_lesion_id']) > 20 else
                                'shrunk' if ch(p['baseline_lesion_id'], p['followup_lesion_id']) < -20 else 'stable'} for p in pairs]}
        else:
            tb_, tf_ = sum(x['volume_mm3'] for x in bl.values()), sum(x['volume_mm3'] for x in fu.values())
            ans = {'baseline_total_volume_mm3': tb_, 'followup_total_volume_mm3': tf_,
                   'burden_direction': 'increase' if tf_ > tb_ else 'decrease' if tf_ < tb_ else 'no_change'}
        return self._R([], json.dumps(ans))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=int, default=0); ap.add_argument('--repeats', type=int, default=3); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--corrections', default=None); ap.add_argument('--mock', action='store_true'); ap.add_argument('--cohort', default='locked')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / ('mock_raw.jsonl' if a.mock else 'generic_stateful_raw.jsonl' if not a.corrections else 'generic_stateful_correction_raw.jsonl')
    pids = [p['patient_id'] for p in G.C.pairs_all() if p.get('cohort') == a.cohort]
    if a.pilot:
        pids = pids[:a.pilot]
    done = set()
    if raw.exists():
        for l in raw.read_text().splitlines():
            j = json.loads(l)
            if (not j['success']) and (j.get('failure_reason') or '').startswith('RuntimeError: api'):
                continue
            done.add((j['patient_id'], j['repeat_id'], j.get('run_tag')))
    if a.corrections:                                                   # one sequence per correction event, repeat 0 (as in the existing test)
        evs = json.loads(Path(a.corrections).read_text())
        tag = lambda e: json.dumps({'id': e['correction_id'], 'pb': e['pb'], 'pf': e['pf']})  # noqa: E731
        jobs = [(e['patient_id'], 0, [e], tag(e)) for e in evs if (e['patient_id'], 0, tag(e)) not in done]
    else:
        jobs = [(pid, r, None, None) for r in range(a.repeats) for pid in pids if (pid, r, None) not in done]
    print('jobs (patient x repeat sequences)', len(jobs), 'model', 'MOCK' if a.mock else G.MODEL, 'prompt sha256', G.PROMPT_HASH[:16], flush=True)
    if a.mock:
        client = MockClient()
    else:
        from openai import OpenAI
        client = OpenAI()
    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(raw, 'a') as f:
        futs = [ex.submit(patient_sequence, client, pid, r, c, tag) for pid, r, c, tag in jobs]
        for fu in as_completed(futs):
            for ep in fu.result():
                f.write(json.dumps(ep) + '\n')
            f.flush(); n += 1
            if n % 25 == 0:
                print(n, '/', len(jobs), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
