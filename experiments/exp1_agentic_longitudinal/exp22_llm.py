"""Experiment 2.2 Code Analysis node: LLM writes code, code computes the answer.

The LLM receives ONLY the clinical question and the patient_state schema (never any patient numbers).
It returns a Python function answer(state); that code is executed locally in a subprocess on the
persisted patient_state.json. Correctness is scored against a deterministic reference implementation
run on the same state, so this isolates LLM code-generation fidelity from perception errors.
OPENAI_API_KEY is read from the environment only.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp22_20260918'
MODEL = 'gpt-5.2'

SCHEMA = """state = {
  "patient_id": str,
  "baseline_lesions": {"<lesion_id>": {"volume_mm3": float, "diameter_mm": float, "centroid_mm": [x,y,z]}},   # keys are strings
  "followup_lesions": {"<lesion_id>": {same fields}},
  "correspondences": [ {"baseline_lesion_id": int|None, "followup_lesion_id": int|None,
                        "status": "persistent" | "disappearing" | "newly_appearing"} ]
}
Rules: persistent = lesion present in both (both ids not None); disappearing = only baseline id; newly_appearing = only follow-up id.
Volumes are in mm3 (1 mL = 1000 mm3). Lesion ids inside correspondences are ints; keys of the lesion dicts are strings."""

QUERIES = {
    'Q0_inventory': ("Compare this patient's baseline and follow-up metastatic lesions and quantify what happened to each lesion.",
                     "Return a list of dicts, one per correspondence row, with keys: baseline_lesion_id, followup_lesion_id, status, "
                     "baseline_volume_mL, followup_volume_mL, volume_change_pct. Volumes in mL (None if the lesion is absent at that time). "
                     "volume_change_pct=(fu-bl)/bl*100 for persistent lesions, else None. Sort rows by (baseline_lesion_id or -1, followup_lesion_id or -1)."),
    'Q1_increased_gt20': ("Which lesions increased in volume by more than 20%?",
                          "Return a sorted list of [baseline_lesion_id, followup_lesion_id] pairs (persistent lesions only, change strictly > +20%)."),
    'Q2_new_count': ("How many new lesions appeared at follow-up?", "Return an int."),
    'Q3_disappeared': ("Which baseline lesions disappeared?", "Return a sorted list of baseline_lesion_id ints."),
}


def reference(state, q):
    bl = {int(k): v for k, v in state['baseline_lesions'].items()}; fu = {int(k): v for k, v in state['followup_lesions'].items()}
    rows = state['correspondences']
    if q == 'Q0_inventory':
        out = []
        for r in rows:
            b, f = r['baseline_lesion_id'], r['followup_lesion_id']
            vb = bl[b]['volume_mm3'] / 1000 if b is not None else None; vf = fu[f]['volume_mm3'] / 1000 if f is not None else None
            out.append({'baseline_lesion_id': b, 'followup_lesion_id': f, 'status': r['status'], 'baseline_volume_mL': vb,
                        'followup_volume_mL': vf, 'volume_change_pct': (vf - vb) / vb * 100 if r['status'] == 'persistent' and vb else None})
        return sorted(out, key=lambda d: (d['baseline_lesion_id'] if d['baseline_lesion_id'] is not None else -1,
                                          d['followup_lesion_id'] if d['followup_lesion_id'] is not None else -1))
    if q == 'Q1_increased_gt20':
        res = []
        for r in rows:
            if r['status'] == 'persistent':
                vb, vf = bl[r['baseline_lesion_id']]['volume_mm3'], fu[r['followup_lesion_id']]['volume_mm3']
                if vb and (vf - vb) / vb * 100 > 20:
                    res.append([r['baseline_lesion_id'], r['followup_lesion_id']])
        return sorted(res)
    if q == 'Q2_new_count':
        return sum(1 for r in rows if r['status'] == 'newly_appearing')
    return sorted(r['baseline_lesion_id'] for r in rows if r['status'] == 'disappearing')


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= 1e-3 * max(1.0, abs(b))
    return a == b


def ask(client, q):
    question, fmt = QUERIES[q]
    prompt = (f"You are the quantitative-analysis node of a medical imaging agent. Question from the clinician: {question}\n\n"
              f"Data available (schema only; you will not see the numbers):\n{SCHEMA}\n\nOutput format: {fmt}\n\n"
              "Write a single Python function `answer(state)` that computes the result from `state` using only the standard library. "
              "Do the arithmetic in code. Return ONLY the code, no explanation, no markdown fences.")
    t0 = time.monotonic()
    r = client.responses.create(model=MODEL, input=prompt, max_output_tokens=2500)
    code = r.output_text.strip()
    if code.startswith('```'):
        code = code.strip('`').split('\n', 1)[1].rsplit('```', 1)[0] if '\n' in code else code
    return code, time.monotonic() - t0


def run_code(code, state_path):
    runner = f"import json,sys\n{code}\nstate=json.load(open(sys.argv[1]))\nprint(json.dumps(answer(state)))\n"
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(runner); p = f.name
    try:
        r = subprocess.run([sys.executable, p, str(state_path)], capture_output=True, text=True, timeout=60)
        if r.returncode:
            return None, r.stderr[-300:]
        return json.loads(r.stdout.strip().splitlines()[-1]), None
    except Exception as e:
        return None, str(e)[:300]
    finally:
        os.unlink(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', nargs='+', default=['S0', 'A', 'B'])
    a = ap.parse_args()
    client = OpenAI()
    # One generated program per query (question+schema fully determine it); each is run on every patient/setting state.
    N = 3  # independent generations per query, to report stability
    codes = {}
    for q in QUERIES:
        codes[q] = [ask(client, q) for _ in range(N)]
        print(q, 'generated', N, 'programs', flush=True)
    (OUT / 'llm_generated_code.json').write_text(json.dumps({q: [{'code': c, 'latency_s': t, 'model': MODEL} for c, t in v] for q, v in codes.items()}, indent=1))
    rows = []
    for s in a.settings:
        for sp in sorted((OUT / f'state_{s}').glob('*.json')):
            st = json.loads(sp.read_text())
            if st.get('status') != 'completed':
                continue
            for q in QUERIES:
                ref = reference(st, q)
                for k, (code, _) in enumerate(codes[q]):
                    got, err = run_code(code, sp)
                    rows.append({'setting': s, 'patient_id': st['patient_id'], 'query': q, 'sample': k, 'executed': err is None,
                                 'correct': err is None and close(got, ref), 'error': err})
    (OUT / 'exp22_llm_eval.json').write_text(json.dumps(rows, indent=1))
    for s in a.settings:
        for q in QUERIES:
            rs = [r for r in rows if r['setting'] == s and r['query'] == q]
            if rs:
                print(s, q, 'n=%d executed=%.2f correct=%.2f' % (len(rs), sum(r['executed'] for r in rs) / len(rs), sum(r['correct'] for r in rs) / len(rs)))


if __name__ == '__main__':
    main()
