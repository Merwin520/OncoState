"""Experiment 2.2 baseline: generic STATE-FREE tool-using agent.

The agent gets the same low-level tools the OncoAgent pipeline is built from (lesion measurement, registration, registered
centroids, the frozen RegisteredHybridMatcher) plus a calculator, and the same frozen definitions of status / change / burden.
It does NOT get a persistent patient-state schema, and every query is a fresh episode: nothing from an earlier query (tool results,
lesion mapping, conclusions) is carried over. The API key is read from the environment only.

Usage:  python exp22b_agent.py [--pilot N] [--repeats R] [--workers W]
"""
import argparse, ast, hashlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import SimpleITK as sitk

import exp22b_common as C
import exp22_run as R

BASE = C.BASE
MODEL = os.environ.get('EXP22B_MODEL', 'gpt-5.2')
RAW = BASE / 'generic_agent_raw.jsonl'
PROMPT_PATH = BASE / 'generic_agent_prompt.txt'
MAX_TURNS = int(os.environ.get('EXP22B_MAX_TURNS', '30'))

SYSTEM = """You are a tool-using analysis agent for serial CT studies (baseline and one follow-up) of a cancer patient.
Lesion masks for both studies already exist; you work through the tools listed below. Lesion ids are integers and are only meaningful
within one timepoint (baseline ids and follow-up ids are unrelated to each other).

Definitions you must use:
- Correspondence: a baseline lesion and a follow-up lesion are a matched pair if they are the same physical lesion; matching is one-to-one.
- Longitudinal status: persistent = present at both timepoints (part of a matched pair); disappearing = baseline lesion without a follow-up counterpart;
  newly_appearing = follow-up lesion without a baseline counterpart.
- Lesion change of a matched pair = (follow-up volume - baseline volume) / baseline volume x 100. enlarged if > +20%, shrunk if < -20%, otherwise stable.
- Patient tumor burden = total volume (mm3) of ALL lesions at a timepoint. Burden direction: increase if the follow-up total is larger than the baseline total,
  decrease if smaller, no_change if equal.

Work only from tool outputs. When you are done, reply with ONE JSON object in the exact format requested and nothing else."""

QUERIES = {
    'Q1': ('Identify the correspondence between baseline and follow-up lesions of this patient.',
           '{"matches": [{"baseline_lesion_id": <int>, "followup_lesion_id": <int>}, ...]}  (matched pairs only)'),
    'Q2': ('Determine the longitudinal status of every lesion of this patient.',
           '{"baseline_status": {"<baseline lesion id>": "persistent" | "disappearing", ...}, "followup_status": {"<follow-up lesion id>": "persistent" | "newly_appearing", ...}}'),
    'Q3': ('Determine, for every persistent lesion of this patient, whether it enlarged, shrank or stayed stable.',
           '{"changes": [{"baseline_lesion_id": <int>, "followup_lesion_id": <int>, "change": "enlarged" | "stable" | "shrunk"}, ...]}'),
    'Q4': ('Determine the overall direction of the patient-level tumor burden (total lesion volume) from baseline to follow-up.',
           '{"baseline_total_volume_mm3": <number>, "followup_total_volume_mm3": <number>, "burden_direction": "increase" | "decrease" | "no_change"}'),
}
PROMPT_TEXT = SYSTEM + '\n\n' + '\n'.join(f'[{k}] {q}\nFormat: {f}' for k, (q, f) in QUERIES.items())
PROMPT_HASH = hashlib.sha256(PROMPT_TEXT.encode()).hexdigest()

TOOLS = [
    {'type': 'function', 'name': 'get_study_info', 'description': 'Number of lesions at baseline and follow-up for a patient.',
     'parameters': {'type': 'object', 'properties': {'patient_id': {'type': 'string'}}, 'required': ['patient_id']}},
    {'type': 'function', 'name': 'list_lesions', 'description': 'Lesion measurements at one timepoint: id, volume (mm3), long-axis diameter (mm), centroid (mm, physical space of that study).',
     'parameters': {'type': 'object', 'properties': {'patient_id': {'type': 'string'}, 'timepoint': {'type': 'string', 'enum': ['baseline', 'followup']}}, 'required': ['patient_id', 'timepoint']}},
    {'type': 'function', 'name': 'register_studies', 'description': 'Rigidly register the follow-up CT to the baseline CT (returns a transform summary). Required before registered_baseline_centroids or match_lesions.',
     'parameters': {'type': 'object', 'properties': {'patient_id': {'type': 'string'}}, 'required': ['patient_id']}},
    {'type': 'function', 'name': 'registered_baseline_centroids', 'description': 'Baseline lesion centroids mapped into follow-up space with the registration transform.',
     'parameters': {'type': 'object', 'properties': {'patient_id': {'type': 'string'}}, 'required': ['patient_id']}},
    {'type': 'function', 'name': 'match_lesions', 'description': 'RegisteredHybridMatcher: one-to-one lesion correspondence from registered distance (30 mm gate), registered overlap and size difference. Returns matched pairs with evidence and the unmatched lesions.',
     'parameters': {'type': 'object', 'properties': {'patient_id': {'type': 'string'}}, 'required': ['patient_id']}},
    {'type': 'function', 'name': 'calculate', 'description': 'Evaluate an arithmetic expression (numbers, + - * / **, sum, len, min, max, abs, round, list comprehensions over literal lists).',
     'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}},
]

PAIRS = {p['patient_id']: p for p in C.pairs_all()}
_cache, _lock = {}, threading.Lock()
ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.ListComp, ast.comprehension, ast.Store,
                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv, ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq, ast.IfExp)
SAFE = {'sum': sum, 'len': len, 'min': min, 'max': max, 'abs': abs, 'round': round}


def cached(key, fn):
    with _lock:
        if key in _cache: return _cache[key]
    val = fn()
    with _lock: _cache[key] = val
    return val


def lesions(pid, tp):
    def f():
        t0 = time.time(); img = sitk.ReadImage(str(R.PATHS['A'](pid, tp))); d = R.describe(img)[1]
        return ({i: dict(volume_mm3=round(v['volume_mm3'], 1), diameter_mm=round(v['diameter_mm'], 1), centroid_mm=[round(x, 1) for x in v['centroid_mm']]) for i, v in d.items()}, time.time() - t0, d)
    return cached(('les', pid, tp), f)


def transform(pid):
    def f():
        t0 = time.time(); p = PAIRS[pid]; tx = R.register(p['baseline_ct'], p['followup_ct'], C.OUT / f'reg/{pid}.tfm'); return tx, time.time() - t0
    return cached(('tx', pid), f)


def matching(pid):
    def f():
        t0 = time.time(); p = PAIRS[pid]; tx, _ = transform(pid)
        bl = sitk.ReadImage(str(R.PATHS['A'](pid, 'baseline'))); fu = sitk.ReadImage(str(R.PATHS['A'](pid, 'followup')))
        rows, _, _ = R.match_fast(bl, fu, tx); return rows, time.time() - t0
    return cached(('match', pid), f)


class Toolbox:
    """One toolbox per episode: records every invocation (name, args, seconds, output ids) for the efficiency and provenance analyses."""
    def __init__(self):
        self.calls = []; self.registered = set(); self.seen_ids = {'baseline': set(), 'followup': set()}

    def run(self, name, args):
        t0 = time.time(); compute = 0.0
        try:
            pid = str(args.get('patient_id', ''))
            if name != 'calculate' and pid not in PAIRS: raise ValueError('unknown patient_id')
            if name == 'get_study_info':
                out = {'patient_id': pid, 'baseline_lesions': len(lesions(pid, 'baseline')[0]), 'followup_lesions': len(lesions(pid, 'followup')[0])}; compute = lesions(pid, 'baseline')[1] + lesions(pid, 'followup')[1]
            elif name == 'list_lesions':
                tp = args['timepoint']; les, compute, _ = lesions(pid, tp); out = [{'id': i, **v} for i, v in sorted(les.items())]; self.seen_ids[tp] |= set(les)
            elif name == 'register_studies':
                tx, compute = transform(pid); self.registered.add(pid); pr = tx.GetParameters()
                out = {'transform': 'rigid Euler3D', 'rotation_deg': [round(float(np.degrees(x)), 2) for x in pr[:3]], 'translation_mm': [round(float(x), 1) for x in pr[3:6]]}
            elif name == 'registered_baseline_centroids':
                if pid not in self.registered: raise ValueError('call register_studies first')
                tx, _ = transform(pid); les, compute, _ = lesions(pid, 'baseline')
                out = [{'baseline_id': i, 'registered_centroid_mm': [round(float(x), 1) for x in tx.TransformPoint(tuple(v['centroid_mm']))]} for i, v in sorted(les.items())]
            elif name == 'match_lesions':
                if pid not in self.registered: raise ValueError('call register_studies first')
                rows, compute = matching(pid)
                out = {'matches': [{'baseline_lesion_id': r['baseline_lesion_id'], 'followup_lesion_id': r['followup_lesion_id'], 'distance_mm': round(r['distance_mm'], 1), 'registered_iou': round(r['registered_iou'], 3), 'size_difference': round(r['relative_size_difference'], 2)} for r in rows if r['status'] == 'persistent'],
                       'unmatched_baseline_ids': [r['baseline_lesion_id'] for r in rows if r['status'] == 'disappearing'], 'unmatched_followup_ids': [r['followup_lesion_id'] for r in rows if r['status'] == 'newly_appearing']}
            elif name == 'calculate':
                expr = str(args.get('expression', ''))
                if len(expr) > 20000 or any(not isinstance(n, ALLOWED_NODES) for n in ast.walk(ast.parse(expr, mode='eval'))): raise ValueError('expression not allowed')
                out = eval(compile(ast.parse(expr, mode='eval'), '<calc>', 'eval'), {'__builtins__': {}}, dict(SAFE))
            else:
                raise ValueError('unknown tool')
            res = {'ok': True, 'result': out}
        except Exception as e:
            res = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
        self.calls.append(dict(name=name, args=args, seconds=round(time.time() - t0, 3), compute_seconds_if_uncached=round(compute, 3), ok=res['ok']))
        return json.dumps(res)


def usage_detail(u):
    g = lambda o, k: (getattr(o, k, 0) or 0) if o is not None else 0
    return dict(input=g(u, 'input_tokens'), cached_input=g(getattr(u, 'input_tokens_details', None), 'cached_tokens'), output=g(u, 'output_tokens'), reasoning=g(getattr(u, 'output_tokens_details', None), 'reasoning_tokens')) if u else dict(input=0, cached_input=0, output=0, reasoning=0)


def parse_json(text):
    try: return json.loads(text[text.index('{'):text.rindex('}') + 1])
    except Exception: return None


def validate(qid, j):
    try:
        if qid == 'Q1': return isinstance(j['matches'], list) and all(isinstance(m['baseline_lesion_id'], int) and isinstance(m['followup_lesion_id'], int) for m in j['matches'])
        if qid == 'Q2': return isinstance(j['baseline_status'], dict) and isinstance(j['followup_status'], dict) and all(v in ('persistent', 'disappearing') for v in j['baseline_status'].values()) and all(v in ('persistent', 'newly_appearing') for v in j['followup_status'].values())
        if qid == 'Q3': return isinstance(j['changes'], list) and all(m['change'] in C.CHANGE and isinstance(m['baseline_lesion_id'], int) and isinstance(m['followup_lesion_id'], int) for m in j['changes'])
        if qid == 'Q4': return j['burden_direction'] in C.BURDEN and isinstance(j['baseline_total_volume_mm3'], (int, float)) and isinstance(j['followup_total_volume_mm3'], (int, float))
    except Exception: return False
    return False


def episode(client, pid, rep, qid, note=None):
    tb = Toolbox(); q, fmt = QUERIES[qid]; t0 = time.time(); tokens = 0; final = ''; fail = None; turns = 0; usage = usage_detail(None)
    items = [{'role': 'developer', 'content': SYSTEM}, {'role': 'user', 'content': f'Patient id: {pid}\nQuery: {q}\nReturn exactly this JSON format: {fmt}' + (f'\n\n{note}' if note else '')}]
    try:
        while turns < MAX_TURNS:
            turns += 1
            r = None
            for attempt in range(4):
                try: r = client.responses.create(model=MODEL, input=items, tools=TOOLS, max_output_tokens=6000); break
                except Exception as e: err = f'{type(e).__name__}: {str(e)[:100]}'; time.sleep(2 + 3 * attempt)
            if r is None: raise RuntimeError('api: ' + err)
            tokens += (r.usage.input_tokens + r.usage.output_tokens) if r.usage else 0
            u = usage_detail(r.usage); usage = {k: usage[k] + u[k] for k in usage}
            items += list(r.output)
            calls = [it for it in r.output if getattr(it, 'type', '') == 'function_call']
            if not calls: final = r.output_text or ''; break
            for it in calls:
                try: args = json.loads(it.arguments)
                except Exception: args = {}
                items.append({'type': 'function_call_output', 'call_id': it.call_id, 'output': tb.run(it.name, args)})
        else:
            fail = 'max_turns'
    except Exception as e:
        fail = f'{type(e).__name__}: {str(e)[:120]}'
    j = parse_json(final) if final else None
    ok = fail is None and j is not None and validate(qid, j)
    if fail is None and not ok: fail = 'invalid_or_missing_json'
    return dict(patient_id=pid, repeat_id=rep, query_id=qid, raw_answer=final, structured_answer=j if ok else None, tool_calls=tb.calls, tools_used=sorted({c['name'] for c in tb.calls}),
                n_tool_calls=len(tb.calls), n_turns=turns, latency_sec=round(time.time() - t0, 2), token_count=tokens, usage=usage, success=bool(ok), failure_reason=None if ok else fail,
                seen_baseline_ids=sorted(tb.seen_ids['baseline']), seen_followup_ids=sorted(tb.seen_ids['followup']))


def main():
    from openai import OpenAI
    ap = argparse.ArgumentParser(); ap.add_argument('--pilot', type=int, default=0); ap.add_argument('--repeats', type=int, default=3); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--raw', default=str(RAW)); ap.add_argument('--cohort', default='locked'); a = ap.parse_args()
    raw = Path(a.raw); PROMPT_PATH.write_text(PROMPT_TEXT)
    pids = [p['patient_id'] for p in C.pairs_all() if p.get('cohort') == a.cohort]
    if a.pilot: pids = pids[:a.pilot]
    done = set()
    if raw.exists():
        for l in raw.read_text().splitlines():
            j = json.loads(l)
            if (not j['success']) and (j.get('failure_reason') or '').startswith('RuntimeError: api'): continue   # API failures are retried
            done.add((j['patient_id'], j['repeat_id'], j['query_id']))
    jobs = [(pid, r, q) for r in range(a.repeats) for pid in pids for q in QUERIES if (pid, r, q) not in done]
    print('jobs', len(jobs), 'done', len(done), 'prompt sha256', PROMPT_HASH[:16], 'model', MODEL, flush=True)
    client = OpenAI(); n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(raw, 'a') as f:
        futs = [ex.submit(episode, client, *j) for j in jobs]
        for fu in as_completed(futs):
            f.write(json.dumps(fu.result()) + '\n'); f.flush(); n += 1
            if n % 50 == 0: print(n, '/', len(jobs), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
