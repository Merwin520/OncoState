"""Experiment 2.3 LLM-only baseline: the model sees the same structured evidence as the rule engine but may not call it.
5 independent inferences per case; every raw answer is stored (llm_raw.jsonl), resumable. The API key is read from the environment only."""
import json, os, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
MODEL = os.environ.get('EXP23_LLM_MODEL', 'gpt-5.2')
REPEATS = int(os.environ.get('EXP23_REPEATS', '5'))
PROMPT = """You are assessing tumor response with RECIST 1.1 from structured evidence. All sums are sums of target-lesion diameters in mm.

{evidence}

Return ONLY a JSON object with these keys:
"computed_change_from_baseline_pct": number or null,
"computed_change_from_nadir_pct": number or null,
"rule_applied": a short statement of the RECIST 1.1 rule(s) you applied, including thresholds,
"response": one of "CR", "PR", "SD", "PD", "NE".
Use "NE" if the evidence is insufficient to assign an overall response category."""
PROMPT_HASH = hashlib.sha256(PROMPT.encode()).hexdigest()[:16]


def fmt(v, unit=''):
    return 'not available' if v is None or (isinstance(v, float) and pd.isna(v)) else f'{v}{unit}'


def evidence_text(r):
    hist = json.loads(r.history_json) if isinstance(r.history_json, str) and r.history_json != 'null' else None
    lines = []
    if hist is None:
        lines.append('Baseline sum: not available')
    else:
        lines.append(f'Baseline sum: {hist[0]} mm')
        prev = hist[1:]
        lines.append('Previous follow-up sums (in order): ' + (', '.join(f'{x} mm' for x in prev) if prev else 'none'))
        lines.append(f'Smallest sum so far before the current visit (nadir, includes baseline): {min(hist)} mm')
    lines.append(f'Current visit sum: {fmt(None if pd.isna(r.current_sum) else float(r.current_sum), " mm")}')
    lines.append(f'Number of target lesions: {r.n_target_lesions}' + ('; at least one target lesion was not measured at the current visit' if r.lesions_missing else ''))
    lines.append('All target lesions disappeared: ' + ('yes' if r.all_disappeared else 'no') + '; all pathologic lymph nodes <10 mm short axis: ' + ('yes' if r.nodes_lt10 else 'no / not applicable'))
    new = r.new_lesion
    lines.append('New lesions: ' + ('not assessed' if (new is None or (isinstance(new, float) and pd.isna(new)) or str(new) in ('None', 'nan')) else ('yes' if str(new) == 'True' else 'no')))
    nt = r.non_target_status
    lines.append('Non-target disease: ' + ('not assessed' if (nt is None or (isinstance(nt, float) and pd.isna(nt)) or str(nt) in ('None', 'nan')) else str(nt)))
    return '\n'.join(lines)


def call(client, cid, rep, prompt):
    t0 = time.time(); last = None
    for attempt in range(4):
        try:
            r = client.responses.create(model=MODEL, input=prompt, max_output_tokens=1500)
            u = r.usage
            return dict(case_id=cid, repeat_id=rep, raw_answer=r.output_text, latency_sec=round(time.time() - t0, 2), token_count=(u.input_tokens + u.output_tokens) if u else None, error=None)
        except Exception as e:
            last = f'{type(e).__name__}: {str(e)[:120]}'; time.sleep(2 + 3 * attempt)
    return dict(case_id=cid, repeat_id=rep, raw_answer='', latency_sec=round(time.time() - t0, 2), token_count=None, error=last)


def main():
    from openai import OpenAI
    limit = int(os.environ.get('EXP23_LIMIT', '0')); workers = int(os.environ.get('EXP23_WORKERS', '16'))
    cases = pd.read_csv(OUT / 'rule_challenge_cases.csv')
    if limit: cases = cases.sample(limit, random_state=1)
    raw = OUT / 'llm_raw.jsonl'; done = set()
    if raw.exists():
        for l in raw.read_text().splitlines():
            j = json.loads(l)
            if not j.get('error'): done.add((j['case_id'], j['repeat_id']))
    jobs = [(r.case_id, k, PROMPT.format(evidence=evidence_text(r))) for r in cases.itertuples() for k in range(REPEATS) if (r.case_id, k) not in done]
    print('jobs', len(jobs), 'already done', len(done), 'prompt hash', PROMPT_HASH, 'model', MODEL, flush=True)
    client = OpenAI(); n = 0
    with ThreadPoolExecutor(workers) as ex, open(raw, 'a') as f:
        futs = [ex.submit(call, client, *j) for j in jobs]
        for fu in as_completed(futs):
            f.write(json.dumps(fu.result()) + '\n'); f.flush(); n += 1
            if n % 200 == 0: print(n, '/', len(jobs), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
