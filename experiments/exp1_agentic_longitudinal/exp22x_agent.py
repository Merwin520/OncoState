"""Experiment 2.2 external comparison: GENERIC tool-using agent (v2, no OncoAgent specialist tools).

The agent receives the SAME visual packet as the direct multimodal models plus generic tools:
  get_lesion_table  - measurements per candidate (volume, diameter, centroid in that scan's own coordinates) from the segmentation used in 2.2
  register_scans    - generic rigid image registration of the two CT studies; returns the transform and the baseline centroids expressed in
                      follow-up coordinates (without it the two coordinate systems cannot be compared at all)
  run_python        - sandboxed Python (numpy, scipy.optimize.linear_sum_assignment, math) with the tables loaded
It has NO OncoAgent matcher, longitudinal-status tool, patient-state lookup, Full predictions or reference correspondence, and it is NOT told
how to match (no mention of centroids or assignment). It decides the computation itself. Backend: the same LLM as elsewhere in the project.
Saved per episode: public tool trace, generated code, tool outputs, final JSON (no private reasoning).
"""
import argparse, json, os, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp22x_vlm as V
import exp22b_common as C
import exp22_run as R
import SimpleITK as sitk

BASE = V.BASE
MODEL = os.environ.get('EXP22X_AGENT_MODEL', 'gpt-5.2')
RAW, ERR = BASE / 'generic_agent_v2_raw.jsonl', BASE / 'generic_agent_v2_api_errors.jsonl'
MAX_TURNS = 20
PY = str(Path(__file__).resolve().parent / 'runtime/bin/python')
INTRO = """You are an analysis agent. You are given serial CT evidence from the same patient as images (see the packet below) and you have the tools listed in the tool definitions.
The candidate lesions have already been located; ids are neutral (B1.. at baseline, F1.. at follow-up) and carry no information about correspondence."""
AGENT_PROMPT = INTRO + '\n\n' + V.PROMPT + '\n\n' + V.DEFINITIONS + '\n\n' + V.SCHEMA_TEXT
import hashlib
PROMPT_SHA = hashlib.sha256(AGENT_PROMPT.encode()).hexdigest()
TOOLS = [
    {'type': 'function', 'name': 'get_lesion_table', 'description': 'Measurements of every candidate lesion of one study: id, volume_mm3, diameter_mm, centroid_mm (in that scan\'s own physical coordinates); also the voxel spacing of the scan. Values are null when no measurement exists.',
     'parameters': {'type': 'object', 'properties': {'timepoint': {'type': 'string', 'enum': ['baseline', 'followup']}}, 'required': ['timepoint']}},
    {'type': 'function', 'name': 'register_scans', 'description': 'Generic rigid registration of the follow-up CT to the baseline CT. Returns the rigid transform parameters and the baseline lesion centroids expressed in follow-up coordinates.',
     'parameters': {'type': 'object', 'properties': {}, 'required': []}},
    {'type': 'function', 'name': 'run_python', 'description': 'Run Python code (numpy as np, math, scipy.optimize.linear_sum_assignment as lsa). Pre-loaded variables: BASELINE and FOLLOWUP (lists of dicts with id, volume_mm3, diameter_mm, centroid_mm) and REGISTERED (dict baseline id -> centroid in follow-up coordinates; empty until register_scans has been called). Print results; stdout is returned.',
     'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}},
]
FORBIDDEN = ('import os', 'import sys', 'subprocess', 'open(', '__import__', 'eval(', 'exec(', 'shutil', 'socket', 'requests', 'pathlib', 'glob', 'input(', 'import importlib', 'os.', 'sys.')
_cache, _lock = {}, threading.Lock()


def world(pid):
    """Agent-visible evidence of one patient (built once)."""
    with _lock:
        if pid in _cache: return _cache[pid]
    key = json.loads((BASE / 'packets' / pid / 'private_key.json').read_text())
    st = json.loads((C.OUT / f'state_A/{pid}.json').read_text()); ref = C.reference(pid, 'A')
    bl = {int(k): v for k, v in st['baseline_lesions'].items()}; fu = {int(k): v for k, v in st['followup_lesions'].items()}
    def table(tp, les, inv):
        rows = []
        for cid, g in sorted(key[tp].items(), key=lambda kv: int(kv[0][1:])):
            m = inv.get(g); v = les.get(m) if m is not None else None
            rows.append(dict(id=cid, volume_mm3=round(v['volume_mm3'], 1) if v else None, diameter_mm=round(v['diameter_mm'], 1) if v else None, centroid_mm=[round(x, 1) for x in v['centroid_mm']] if v else None))
        return rows
    p = next(x for x in C.pairs_all() if x['patient_id'] == pid); tx = R.register(p['baseline_ct'], p['followup_ct'], C.OUT / f'reg/{pid}.tfm')
    B, F = table('baseline', bl, ref['inv_b']), table('followup', fu, ref['inv_f'])
    reg = {r['id']: [round(float(x), 1) for x in tx.TransformPoint(tuple(r['centroid_mm']))] for r in B if r['centroid_mm']}
    sp = {}
    for tp, path in (('baseline', p['baseline_ct']), ('followup', p['followup_ct'])):
        rd = sitk.ImageFileReader(); rd.SetFileName(path); rd.ReadImageInformation(); sp[tp] = [round(x, 3) for x in rd.GetSpacing()]
    pr = tx.GetParameters(); out = dict(B=B, F=F, reg=reg, spacing=sp, rot=[round(float(np.degrees(x)), 2) for x in pr[:3]], trans=[round(float(x), 1) for x in pr[3:6]], key=key)
    with _lock: _cache[pid] = out
    return out


def run_python(w, code, registered):
    if any(f in code for f in FORBIDDEN): return json.dumps({'ok': False, 'error': 'code rejected: forbidden operation'})
    with tempfile.TemporaryDirectory() as td:
        data = dict(baseline=w['B'], followup=w['F'], registered=w['reg'] if registered else {})
        (Path(td) / 'd.json').write_text(json.dumps(data))
        pre = "import json, math\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment as lsa\n_d = json.load(open('d.json'))\nBASELINE, FOLLOWUP, REGISTERED = _d['baseline'], _d['followup'], _d['registered']\n"
        (Path(td) / 'run.py').write_text(pre + code)
        try:
            r = subprocess.run([PY, 'run.py'], cwd=td, capture_output=True, text=True, timeout=25, env={'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1'})
            return json.dumps({'ok': r.returncode == 0, 'stdout': r.stdout[-5000:], 'stderr': r.stderr[-1500:]})
        except subprocess.TimeoutExpired: return json.dumps({'ok': False, 'error': 'timeout'})


def tool(w, state, name, args):
    t0 = time.time()
    try:
        if name == 'get_lesion_table': out = {'lesions': w['B'] if args['timepoint'] == 'baseline' else w['F'], 'voxel_spacing_mm': w['spacing'][args['timepoint']]}; res = json.dumps({'ok': True, 'result': out})
        elif name == 'register_scans': state['registered'] = True; res = json.dumps({'ok': True, 'result': dict(transform='rigid Euler3D', rotation_deg=w['rot'], translation_mm=w['trans'], baseline_centroids_in_followup_coordinates_mm=w['reg'])})
        elif name == 'run_python': res = run_python(w, str(args.get('code', '')), state.get('registered', False))
        else: res = json.dumps({'ok': False, 'error': 'unknown tool'})
    except Exception as e: res = json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'})
    return res, dict(name=name, args=args, output=res[:3000], seconds=round(time.time() - t0, 2))


def episode(client, pid, rep):
    w = world(pid); inp, pub = V.build_input(pid); content = inp[0]['content']; content[0] = {'type': 'input_text', 'text': AGENT_PROMPT}
    items = [{'role': 'user', 'content': content}]; state = {}; trace = []; tokens = 0; usage = V.usage_detail(None); t0 = time.time(); text = ''; turns = 0; api_fail = None
    while turns < MAX_TURNS:
        turns += 1; r = None
        for attempt in range(4):
            try: r = client.responses.create(model=MODEL, input=items, tools=TOOLS, max_output_tokens=20000); break
            except Exception as e:
                msg = str(e); quota = 'insufficient_quota' in msg or 'no credits' in msg.lower()
                with V._lock, open(ERR, 'a') as f: f.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(), model_id=MODEL, patient_id=pid, repeat_id=rep, error_type=type(e).__name__, http_status=getattr(e, 'status_code', None), error_message=msg[:200])) + '\n')
                if quota: api_fail = 'quota'; break
                time.sleep(2 ** (attempt + 1) * 2)
        if r is None: return dict(patient_id=pid, repeat_id=rep, api_success=False, status=api_fail or 'api_failed')
        tokens += (r.usage.input_tokens + r.usage.output_tokens) if r.usage else 0; usage = V.usage_add(usage, V.usage_detail(r.usage)); items += list(r.output)
        calls = [it for it in r.output if getattr(it, 'type', '') == 'function_call']
        if not calls: text = r.output_text or ''; break
        for it in calls:
            try: a = json.loads(it.arguments)
            except Exception: a = {}
            res, tr = tool(w, state, it.name, a); trace.append(tr); items.append({'type': 'function_call_output', 'call_id': it.call_id, 'output': res})
    j = V.parse(text, pub) if text else None
    return dict(patient_id=pid, repeat_id=rep, api_success=True, valid=j is not None, structured_answer=j, raw_answer=text, tool_trace=trace, tools_used=sorted({t['name'] for t in trace}), n_tool_calls=len(trace),
                n_turns=turns, latency_sec=round(time.time() - t0, 1), token_count=tokens, usage=usage, status='ok' if j is not None else ('max_turns' if not text else 'invalid_output'), model_id=MODEL, timestamp=datetime.now(timezone.utc).isoformat())


def main():
    global RAW
    from openai import OpenAI
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', default='locked'); ap.add_argument('--pids', default=''); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--raw', default=str(RAW)); a = ap.parse_args(); RAW = Path(a.raw)
    pids = a.pids.split(',') if a.pids else [p['patient_id'] for p in C.pairs_all() if p.get('cohort') == a.cohort]
    if a.limit and not a.pids: pids = pids[:a.limit]
    done = set()
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            j = json.loads(l)
            if j.get('api_success'): done.add((j['patient_id'], j['repeat_id']))
    jobs = [(p, r) for r in range(a.repeats) for p in pids if (p, r) not in done and (BASE / 'packets' / p / 'public_packet.json').exists()]
    print(f'planned {len(jobs)} (done {len(done)}); prompt sha256 {PROMPT_SHA[:16]}; model {MODEL}', flush=True); client = OpenAI(); n = 0; stop = threading.Event()
    def g(p, r):
        if stop.is_set(): return dict(api_success=False, status='skipped')
        res = episode(client, p, r)
        if res.get('status') == 'quota': stop.set()
        return res
    with ThreadPoolExecutor(a.workers) as ex, open(RAW, 'a') as f:
        futs = [ex.submit(g, *j) for j in jobs]
        for fu in as_completed(futs):
            res = fu.result()
            if not res.get('api_success'): continue                                          # infrastructure failure: retried on re-run, never stored as a result
            f.write(json.dumps(res) + '\n'); f.flush(); n += 1
            if n % 20 == 0: print(n, '/', len(jobs), flush=True)
    print('finished', n, 'quota stop' if stop.is_set() else '')


if __name__ == '__main__':
    main()
