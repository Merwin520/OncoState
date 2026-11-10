"""Medical open-weight VLM rows of the external comparison table (patient-level bootstrap, same evaluation as exp22x_main)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp22x_eval as X, exp22x_main as M
B = X.BASE; NAMES = {'llava_med': 'Direct medical VLM (LLaVA-Med v1.5 7B)', 'medgemma': 'Direct medical VLM (MedGemma 4B-it)'}
rows = []; det, refs, keys = M.load_all()
raw = [json.loads(l) for l in open(B / 'medical_vlm_raw.jsonl')]
for mk in NAMES:
    rs = {r['patient_id']: r for r in raw if r['model_key'] == mk and r.get('run_success')}
    if not rs: continue
    pids = sorted(p for p in rs if p in refs); V = []
    for p in pids:
        V.append(X.evaluate(refs[p], X.packet_pred(rs[p]['structured_answer'], keys[p])))
    V = np.array(V); pt, bt, _ = X.bootstrap({mk: V}, {mk: np.ones(len(pids), bool)}, pids, nb=2000)
    row = dict(method=NAMES[mk], n_evaluated=len(pids), n_valid_output=sum(bool(rs[p].get('valid')) for p in pids))
    for k, lab in (('match_f1', 'matching_f1'), ('change_accuracy', 'change_accuracy'), ('burden_accuracy', 'burden_accuracy')):
        lo, hi = np.nanpercentile(bt[mk][k], [2.5, 97.5]); row[lab] = round(pt[mk][k], 3); row[lab + '_ci95'] = f'[{lo:.3f}, {hi:.3f}]'
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv(B / 'medical_vlm_metrics.csv', index=False); print(df.to_string())
