"""Experiment 2.2 external comparison: DIRECT multimodal models on the standardized visual packets.

Roles are fixed: the model only looks at the packet images and answers. It has no tools and no access to any OncoAgent matcher, predicted
correspondence, status tool, persistent state, reference correspondence or reference response label. Every model receives the identical
frozen prompt and the identical packet. Each patient x model is run three times; repeat 0 is the primary analysis.

Robust checkpointing: every finished inference is appended immediately to direct_vlm_raw.jsonl (key = model, patient, repeat) and skipped
on re-run. API infrastructure failures (quota, credits, rate limit, timeout, server error) go to direct_vlm_api_errors.jsonl and are NEVER
counted as model failures. An invalid structured answer after a SUCCESSFUL request is a task failure (one schema-repair retry is allowed;
it may only ask for re-formatting, not re-interpretation of the images).
"""
import argparse, base64, hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent / 'outputs/exp22_external_final'
MODELS = {'A_strongest': 'gpt-5.5', 'B_medium': 'gpt-5.2', 'C_light': 'gpt-5.4-mini'}
REPEATS = 3
RAW, ERR = BASE / 'direct_vlm_raw.jsonl', BASE / 'direct_vlm_api_errors.jsonl'

PROMPT = """You are given serial CT evidence from the same patient.
Baseline lesions and follow-up candidate lesions are shown
with axial, coronal, and sagittal views.

Your tasks are:

1. Match each baseline lesion to the most likely corresponding
follow-up lesion. If no convincing correspondence exists,
return no_match.

2. For matched lesions, determine whether the lesion has
increased, decreased, or remained stable.

3. Determine the overall direction of tumor burden across
the provided lesions.

Use only the supplied images and metadata.
Do not assume that IDs correspond across scans.
Return only the required JSON schema."""

SCHEMA_TEXT = """Required JSON schema (return exactly this structure and nothing else):
{
  "matches": [
    {"baseline_id": "B1", "followup_id": "F2" or "no_match", "change": "increased" | "stable" | "decreased" | "uncertain" | "not_applicable"}
  ],
  "burden_direction": "increased" | "stable" | "decreased" | "uncertain"
}
Include exactly one entry for every baseline lesion. Use "not_applicable" for the change of a baseline lesion whose followup_id is "no_match"."""
DEFINITIONS = """Definitions of the labels (the same definitions are used for every method):
- A matched lesion has "increased" if its volume grew by more than 20%, "decreased" if its volume shrank by more than 20%, and is "stable" otherwise.
- Tumor burden is the total volume of all lesions at a visit. Its direction is "increased" if the follow-up total is larger than the baseline total and
  "decreased" if it is smaller; use "stable" only if the two totals are essentially equal.
- Use "uncertain" only when the evidence really does not allow a judgement."""
PROMPT_FULL = PROMPT + '\n\n' + DEFINITIONS + '\n\n' + SCHEMA_TEXT
PROMPT_SHA = hashlib.sha256(PROMPT_FULL.encode()).hexdigest()
REPAIR = "Your previous answer did not follow the required JSON schema. Re-format your previous answer into exactly the required schema. Do not re-interpret any image; only change the format. Return only the JSON."
_lock, _stop = threading.Lock(), set()


def usage_detail(u):
    """Full token accounting of one API response (for exact cost reporting)."""
    if u is None: return dict(input=0, cached_input=0, output=0, reasoning=0)
    g = lambda o, k: (getattr(o, k, 0) or 0) if o is not None else 0
    return dict(input=g(u, 'input_tokens'), cached_input=g(getattr(u, 'input_tokens_details', None), 'cached_tokens'), output=g(u, 'output_tokens'), reasoning=g(getattr(u, 'output_tokens_details', None), 'reasoning_tokens'))


def usage_add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in ('input', 'cached_input', 'output', 'reasoning')}


def b64(path):
    return 'data:image/png;base64,' + base64.b64encode((BASE / path).read_bytes()).decode()


def build_input(pid):
    pub = json.loads((BASE / 'packets' / pid / 'public_packet.json').read_text()); content = [{'type': 'input_text', 'text': PROMPT_FULL}]
    for tp, title in (('baseline', 'BASELINE study (earlier scan)'), ('followup', 'FOLLOW-UP study (later scan)')):
        p = pub[tp]; ids = ', '.join(p['candidate_ids'])
        content.append({'type': 'input_text', 'text': f'\n{title}. Candidate lesion ids: {ids}.\nCoronal overview with candidate positions marked (R = patient right on the left of the image):'})
        content.append({'type': 'input_image', 'image_url': b64(p['overview']), 'detail': 'high'})
        content.append({'type': 'input_text', 'text': f'{title}: lesion-centred views. Each tile is labelled with its candidate id and shows axial | coronal | sagittal panels of 80 mm, centred on the candidate position (20 mm scale bar in the axial panel). Pages follow.'})
        for pg in p['pages']: content.append({'type': 'input_image', 'image_url': b64(pg), 'detail': 'high'})
    return [{'role': 'user', 'content': content}], pub


def parse(text, pub):
    try:
        j = json.loads(text[text.index('{'):text.rindex('}') + 1]); ms = j['matches']; bd = j['burden_direction']
        assert isinstance(ms, list) and bd in ('increased', 'stable', 'decreased', 'uncertain')
        bids, fids = set(pub['baseline']['candidate_ids']), set(pub['followup']['candidate_ids']) | {'no_match'}
        for m in ms: assert m['baseline_id'] in bids and m['followup_id'] in fids and m['change'] in ('increased', 'stable', 'decreased', 'uncertain', 'not_applicable')
        return j
    except Exception:
        return None


def call(client, model, pid, rep):
    inp, pub = build_input(pid); t0 = time.time(); tokens = 0; repaired = False; usage = usage_detail(None)
    def once(inp_):
        for attempt in range(4):
            try:
                r = client.responses.create(model=model, input=inp_, max_output_tokens=20000); return r, None
            except Exception as e:
                name, msg = type(e).__name__, str(e)
                status = getattr(e, 'status_code', None)
                quota = 'insufficient_quota' in msg or 'no credits' in msg.lower() or 'credit' in msg.lower() and 'exhaust' in msg.lower()
                with _lock, open(ERR, 'a') as f:
                    f.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(), model_id=model, patient_id=pid, repeat_id=rep, error_type=name, http_status=status, error_message=msg[:200].replace(os.environ.get('OPENAI_API_KEY', '@@'), '***'))) + '\n')
                if quota: _stop.add(model); return None, 'quota'
                if attempt < 3: time.sleep(2 ** (attempt + 1) * 2)
        return None, 'api_failed'
    r, err = once(inp)
    if r is None: return dict(model_id=model, patient_id=pid, repeat_id=rep, status=err, api_success=False)
    tokens += (r.usage.input_tokens + r.usage.output_tokens) if r.usage else 0; usage = usage_add(usage, usage_detail(r.usage)); text = r.output_text or ''; api_model = r.model; j = parse(text, pub)
    if j is None:
        repaired = True; r2, err2 = once([{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Previous answer:\n' + text[:12000] + '\n\n' + REPAIR + '\n\n' + SCHEMA_TEXT}]}])
        if r2 is None: return dict(model_id=model, patient_id=pid, repeat_id=rep, status=err2, api_success=False)
        tokens += (r2.usage.input_tokens + r2.usage.output_tokens) if r2.usage else 0; usage = usage_add(usage, usage_detail(r2.usage)); text = r2.output_text or ''; j = parse(text, pub)
    return dict(model_id=model, api_model=api_model, patient_id=pid, repeat_id=rep, api_success=True, valid=j is not None, schema_repair_used=repaired, structured_answer=j, raw_answer=text,
                latency_sec=round(time.time() - t0, 1), token_count=tokens, usage=usage, status='ok' if j is not None else 'invalid_output', timestamp=datetime.now(timezone.utc).isoformat())


def main():
    global RAW
    from openai import OpenAI
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', default='locked'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--repeats', type=int, default=REPEATS)
    ap.add_argument('--models', default=','.join(MODELS)); ap.add_argument('--raw', default=str(RAW)); ap.add_argument('--workers', type=int, default=10); ap.add_argument('--pids', default=''); a = ap.parse_args()
    RAW = Path(a.raw)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import exp22b_common as C
    pids = a.pids.split(',') if a.pids else [p['patient_id'] for p in C.pairs_all() if p.get('cohort') == a.cohort]
    if a.limit and not a.pids: pids = pids[:a.limit]
    done = set()
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            j = json.loads(l)
            if j.get('api_success'): done.add((j['model_id'], j['patient_id'], j['repeat_id']))
    models = [MODELS[m] for m in a.models.split(',')]
    jobs = [(m, p, r) for r in range(a.repeats) for m in models for p in pids if (m, p, r) not in done and (BASE / 'packets' / p / 'public_packet.json').exists()]
    print(f'planned {len(jobs)} (already done {len(done)}); prompt sha256 {PROMPT_SHA[:16]}', flush=True)
    client = OpenAI(); n = 0
    def guarded(m, p, r):
        if m in _stop: return dict(model_id=m, patient_id=p, repeat_id=r, status='skipped_quota_stop', api_success=False)
        return call(client, m, p, r)
    with ThreadPoolExecutor(a.workers) as ex, open(RAW, 'a') as f:
        futs = [ex.submit(guarded, *j) for j in jobs]
        for fu in as_completed(futs):
            res = fu.result()
            if res.get('status') in ('quota', 'skipped_quota_stop', 'api_failed'): continue      # infrastructure failure: not stored as a result (see api_errors), retried on re-run
            f.write(json.dumps(res) + '\n'); f.flush(); n += 1
            if n % 50 == 0: print(n, '/', len(jobs), flush=True)
    print('finished', n, 'stopped models:', sorted(_stop))


if __name__ == '__main__':
    main()
