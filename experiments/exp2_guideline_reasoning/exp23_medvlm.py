"""Experiment 2.3 open-weight medical-VLM baseline: LLaVA-Med and MedGemma see the SAME structured text evidence as the
LLM-only baseline (exp23_llm_only.py, identical prompt) and answer with no tool/rule engine. Text-only (no images) — the RECIST
evidence here is numeric, not visual. Local inference (anatomy_runtime), greedy decoding, one run per case (deterministic).
Covers both the synthetic rule-challenge main subset (rule_challenge_cases.csv, 'main') and the real HCUCH cohort
(real_hcuch_cases.csv, target-lesion response only). Checkpointed to medvlm_raw.jsonl (key = model, case_id, source).
Usage: CUDA_VISIBLE_DEVICES=0 python exp23_medvlm.py --model llava_med|medgemma [--source main|real] [--limit N]
"""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'exp1_agentic_longitudinal'))
import exp22x_medvlm as MV                       # Runner (model loading) shared with Experiment 2.2
import exp23_llm_only as L                        # frozen PROMPT + evidence_text()
import exp23_systems as S                          # parse_llm(), RL.RESPONSES

RAW = OUT / 'medvlm_raw.jsonl'
MODEL_ID = {'llava_med': 'llava-med-v1.5-mistral-7b (HF port: chaoyinshe/llava-med-v1.5-mistral-7b-hf)', 'medgemma': 'google/medgemma-4b-it'}


def real_evidence_text(r):
    """Same shape as exp23_llm_only.evidence_text(), built from real_hcuch_cases.csv columns (target-lesion evidence only)."""
    lines = [f'Baseline sum: {r.baseline_sum} mm', 'Previous follow-up sums (in order): not individually available',
             f'Smallest sum so far before the current visit (nadir, includes baseline): {r.prior_nadir_sum} mm',
             f'Current visit sum: {r.followup_sum} mm', f'Number of target lesions: {r.n_target_lesions}' + ('' if r.evidence_complete_target else '; at least one target lesion was not measured at the current visit'),
             'All target lesions disappeared: no; all pathologic lymph nodes <10 mm short axis: not applicable', 'New lesions: not assessed', 'Non-target disease: not assessed']
    return '\n'.join(lines)


def jobs(source, limit):
    if source == 'main':
        cases = pd.read_csv(OUT / 'rule_challenge_cases.csv'); cases = cases[cases.subset == 'main']
        if limit: cases = cases.sample(limit, random_state=1)
        return [(r.case_id, L.PROMPT.format(evidence=L.evidence_text(r)), r.expected_response, r.expected_rule_ids) for r in cases.itertuples()]
    cases = pd.read_csv(OUT / 'real_hcuch_cases.csv')
    if limit: cases = cases.iloc[:limit]
    return [(f'{r.patient_id}_{r.study_pair}', L.PROMPT.format(evidence=real_evidence_text(r)), r.reference_response, 'target-lesion') for r in cases.itertuples()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=list(MODEL_ID)); ap.add_argument('--source', default='main', choices=['main', 'real']); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    todo_all = jobs(a.source, a.limit); done = set()
    if RAW.exists():
        for l in RAW.read_text().splitlines(): j = json.loads(l); done.add((j['model_key'], j['source'], j['case_id']))
    todo = [j for j in todo_all if (a.model, a.source, j[0]) not in done]
    print(f'{a.model}/{a.source}: {len(todo)} cases to run ({len(todo_all) - len(todo)} done); prompt hash {L.PROMPT_HASH}', flush=True)
    if not todo: return
    r = MV.Runner(a.model); n = 0
    with open(RAW, 'a') as f:
        for cid, prompt, exp, exp_ids in todo:
            t0 = time.time()
            try:
                text, n_in = r.generate_text(prompt); pred, obj = S.parse_llm(text)
                row = dict(model_key=a.model, model_id=MODEL_ID[a.model], source=a.source, case_id=cid, expected_response=exp, expected_rule_ids=exp_ids, predicted_response=pred, correct=pred == exp,
                           raw_answer=text, structured=obj, input_tokens=n_in, latency_sec=round(time.time() - t0, 1), decoding='greedy', run_success=True)
            except Exception as e:
                row = dict(model_key=a.model, source=a.source, case_id=cid, run_success=False, status=f'{type(e).__name__}: {str(e)[:150]}')
            f.write(json.dumps(row) + '\n'); f.flush(); n += 1
            if n % 100 == 0: print(n, '/', len(todo), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
