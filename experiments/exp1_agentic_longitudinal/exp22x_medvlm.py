"""Experiment 2.2 external comparison: open-weight MEDICAL vision-language models (LLaVA-Med, MedGemma) run locally on the same standardized packets.

Same role as the direct GPT models: the model only looks at the packet images and answers with the same frozen prompt (including the shared label
definitions) and the same JSON schema; no tools, no matcher, no state, no reference. Greedy decoding (deterministic), so one run per patient.
Adaptations forced by the architectures (identical for every patient, decided before evaluation):
  * pages are padded to a square (black) before the model's own resizing, so tiles are not cropped or stretched;
  * LLaVA-Med (single-image, 336 px, 576 tokens per image) receives at most 6 images: both overviews and up to two lesion pages per study
    (first 16 candidates per study); MedGemma receives every page.
An invalid structured answer gets one text-only schema-repair attempt (re-formatting only), as for the GPT models.
Usage: CUDA_VISIBLE_DEVICES=0 python exp22x_medvlm.py --model llava_med|medgemma [--cohort locked|dev] [--pids a,b] [--limit N]
"""
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp22x_vlm as V
from PIL import Image

BASE = V.BASE
MODELS_DIR = Path(__file__).resolve().parent / 'models/external_vlm'
SPEC = {'llava_med': dict(path=MODELS_DIR / 'llava-med-v1.5-mistral-7b-hf', model_id='llava-med-v1.5-mistral-7b (HF port: chaoyinshe/llava-med-v1.5-mistral-7b-hf)', max_images=6, max_pages_per_study=2, size=336),
        'medgemma': dict(path=MODELS_DIR / 'medgemma-4b-it', model_id='google/medgemma-4b-it', max_images=None, max_pages_per_study=None, size=896)}
RAW = BASE / 'medical_vlm_raw.jsonl'
MAX_NEW = 1500


def pad_square(im, size):
    im = im.convert('RGB'); s = max(im.size); bg = Image.new('RGB', (s, s), (0, 0, 0)); bg.paste(im, ((s - im.width) // 2, (s - im.height) // 2)); return bg.resize((size, size), Image.BICUBIC)


def build(pid, spec):
    """Same text/image order as the GPT packet, as a list of ('text', str) / ('image', PIL) parts."""
    pub = json.loads((BASE / 'packets' / pid / 'public_packet.json').read_text()); parts = [('text', V.PROMPT_FULL)]; truncated = False; n_img = 0
    for tp, title in (('baseline', 'BASELINE study (earlier scan)'), ('followup', 'FOLLOW-UP study (later scan)')):
        p = pub[tp]; parts.append(('text', f'\n{title}. Candidate lesion ids: {", ".join(p["candidate_ids"])}.\nCoronal overview with candidate positions marked (R = patient right on the left of the image):'))
        parts.append(('image', pad_square(Image.open(BASE / p['overview']), spec['size']))); n_img += 1
        parts.append(('text', f'{title}: lesion-centred views. Each tile is labelled with its candidate id and shows axial | coronal | sagittal panels of 80 mm, centred on the candidate position (20 mm scale bar in the axial panel). Pages follow.'))
        pages = p['pages'] if spec['max_pages_per_study'] is None else p['pages'][:spec['max_pages_per_study']]; truncated |= len(pages) < len(p['pages'])
        for pg in pages: parts.append(('image', pad_square(Image.open(BASE / pg), spec['size']))); n_img += 1
    return pub, parts, truncated, n_img


class Runner:
    def __init__(self, name):
        import torch
        from transformers import AutoProcessor
        self.name, self.spec, self.torch = name, SPEC[name], torch
        path = str(self.spec['path']); self.proc = AutoProcessor.from_pretrained(path)
        if name == 'llava_med':
            from transformers import LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16, device_map='cuda')
        else:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda')
        self.model.eval()

    def generate(self, parts):
        imgs = [x for k, x in parts if k == 'image']
        if self.name == 'llava_med':
            content = []
            for k, x in parts: content.append({'type': 'image'} if k == 'image' else {'type': 'text', 'text': x})
            text = self.proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True)
            inputs = self.proc(images=imgs, text=text, return_tensors='pt').to('cuda', self.torch.float16)
        else:
            content = [({'type': 'image', 'image': x} if k == 'image' else {'type': 'text', 'text': x}) for k, x in parts]
            inputs = self.proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to('cuda', self.torch.bfloat16)
        n_in = inputs['input_ids'].shape[-1]
        with self.torch.inference_mode(): out = self.model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        return self.proc.decode(out[0][n_in:], skip_special_tokens=True), n_in

    def generate_text(self, prompt, max_new=300):
        """Text-only generation (no images) — used by the Experiment 2.3 guideline-reasoning baseline."""
        if self.name == 'llava_med':
            text = self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}], add_generation_prompt=True)
            inputs = self.proc(text=text, return_tensors='pt').to('cuda')
        else:
            inputs = self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to('cuda')
        n_in = inputs['input_ids'].shape[-1]
        with self.torch.inference_mode(): out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return self.proc.decode(out[0][n_in:], skip_special_tokens=True), n_in

    def repair(self, prev):
        msg = 'Previous answer:\n' + prev[:6000] + '\n\n' + V.REPAIR + '\n\n' + V.SCHEMA_TEXT
        if self.name == 'llava_med':
            text = self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': msg}]}], add_generation_prompt=True); inputs = self.proc(text=text, return_tensors='pt').to('cuda')
        else:
            inputs = self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': msg}]}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to('cuda')
        n_in = inputs['input_ids'].shape[-1]
        with self.torch.inference_mode(): out = self.model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        return self.proc.decode(out[0][n_in:], skip_special_tokens=True)


def main():
    global RAW
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=list(SPEC)); ap.add_argument('--cohort', default='locked'); ap.add_argument('--pids', default=''); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--raw', default=str(RAW)); a = ap.parse_args()
    RAW = Path(a.raw); import exp22b_common as C
    pids = a.pids.split(',') if a.pids else [p['patient_id'] for p in C.pairs_all() if p.get('cohort') == a.cohort]
    if a.limit and not a.pids: pids = pids[:a.limit]
    spec = SPEC[a.model]; done = set()
    if RAW.exists():
        for l in RAW.read_text().splitlines(): j = json.loads(l); done.add((j['model_key'], j['patient_id']))
    todo = [p for p in pids if (a.model, p) not in done and (BASE / 'packets' / p / 'public_packet.json').exists()]
    print(f'{a.model}: {len(todo)} patients to run ({len(done)} done); prompt sha256 {V.PROMPT_SHA[:16]}', flush=True)
    r = Runner(a.model); n = 0
    with open(RAW, 'a') as f:
        for pid in todo:
            t0 = time.time()
            try:
                pub, parts, trunc, n_img = build(pid, spec); text, n_in = r.generate(parts); j = V.parse(text, pub); repaired = False
                if j is None: repaired = True; text = r.repair(text); j = V.parse(text, pub)
                row = dict(model_key=a.model, model_id=spec['model_id'], patient_id=pid, repeat_id=0, run_success=True, valid=j is not None, schema_repair_used=repaired, structured_answer=j, raw_answer=text, latency_sec=round(time.time() - t0, 1),
                           input_tokens=n_in, n_images=n_img, packet_truncated=bool(trunc), status='ok' if j is not None else 'invalid_output', decoding='greedy', timestamp=datetime.now(timezone.utc).isoformat())
            except Exception as e:
                row = dict(model_key=a.model, model_id=spec['model_id'], patient_id=pid, repeat_id=0, run_success=False, status=f'runtime_error: {type(e).__name__}: {str(e)[:150]}', timestamp=datetime.now(timezone.utc).isoformat())
                import torch; torch.cuda.empty_cache()
            f.write(json.dumps(row) + '\n'); f.flush(); n += 1
            if n % 10 == 0: print(n, '/', len(todo), flush=True)
    print('finished', n)


if __name__ == '__main__':
    main()
