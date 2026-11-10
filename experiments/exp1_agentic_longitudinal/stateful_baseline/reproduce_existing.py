#!/usr/bin/env python3
"""Sanity step 1 (no API calls): compute the deterministic Exp 2.2 methods once (cached to outputs/stateful_baseline/det_cache.pkl),
recompute the archived comparison with the unchanged evaluation code and compare with baseline_metrics.csv, and write the 60 correction
events of the existing retention test (exp22b_retention.events, fixed seed) for the agent runs. Writes nothing into the original folders."""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exp22b_run as RUN  # noqa: E402
import exp22b_retention as RET  # noqa: E402

OUT = HERE.parents[2] / 'outputs/stateful_baseline'
OUT.mkdir(parents=True, exist_ok=True)
det, bad = RUN.deterministic()
(OUT / 'det_cache.pkl').write_bytes(pickle.dumps((det, bad)))
A = RUN.analysis(det, RUN.agent_records())
new = A['df'][['method', 'metric', 'estimate', 'ci95_lo', 'ci95_hi', 'n_patients']].copy()
new['method'] = new.method.astype(str)
old = pd.read_csv(RUN.BASE / 'baseline_metrics.csv')
m = old.merge(new, on=['method', 'metric'], suffixes=('_archived', '_recomputed'))
m['abs_diff'] = (m.estimate_archived - m.estimate_recomputed).abs()
m.to_csv(OUT / 'reproduction_check.csv', index=False)
evs, n_all = RET.events(det)
corr = [dict(correction_id=f'C{i:02d}', patient_id=e['pid'], pb=int(e['pb']), pf=int(e['pf']),
             statement=f"Verified clinician correction (apply it): baseline lesion {e['pb']} and follow-up lesion {e['pf']} are the same lesion, "
                       f"i.e. a matched pair.")
        for i, e in enumerate(evs)]
(OUT / 'correction_events.json').write_text(json.dumps(corr, indent=1))
old_ev = pd.read_csv(RUN.BASE.parent / 'exp22_external_final/edit_retention.csv').query("system == 'Full OncoAgent' and query == 'Q1'")
same = sorted(zip(old_ev.patient_id, old_ev.pb, old_ev.pf)) == sorted((c['patient_id'], c['pb'], c['pf']) for c in corr)
print('failed deterministic evaluations:', len(bad), '| patients:', len(A['patients']), '| rows compared:', len(m), '| max |diff|:', m.abs_diff.max())
print('correction events:', len(corr), 'of', n_all, 'real matching errors; identical to archived retention events:', same)
print(m[m.metric.isin(['match_f1', 'status_weighted_f1', 'status_kappa', 'burden_accuracy'])][['method', 'metric', 'estimate_archived',
                                                                                                 'estimate_recomputed']].to_string(index=False))
