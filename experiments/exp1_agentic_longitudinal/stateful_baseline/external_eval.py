#!/usr/bin/env python3
"""Score the two same-model agents (Qwen3.5-9B stateless / generic stateful) with the UNCHANGED external Experiment 2.2 evaluator
(exp22x_eval: reference-lesion space, change = increased/stable/decreased over all reference persistent lesions, patient bootstrap
NB=5000, seed 20260921), so that Direct VLM, both agents and OncoState on one definition.
Agent answers -> reference lesions exactly as exp22x_eval.det_pred does for the deterministic pipeline (frozen IoU>0.1 mapping):
Q1 pairs, Q3 change labels, Q4 burden direction. Per patient the sufficient statistics are averaged over the 3 repeats.
Sanity: Full OncoState must reproduce exp22_external_final/external_metrics.csv exactly, otherwise the script stops.
Writes outputs/stateful_baseline/external_metrics_same_definition.csv and external_paired_same_definition.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import analyze_stateful as A  # noqa: E402
import exp22b_run as RUN  # noqa: E402
import exp22x_eval as X  # noqa: E402

OUT = A.OUT
NAMES = {A.STATELESS: 'Stateless tool agent', A.STATEFUL: 'Generic stateful tool agent'}


def to_ref(ref, pred):
    pairs, change = [], {}
    for a, b in pred['pairs']:
        pairs.append((ref['mb'].get(a), ref['mf'].get(b)))
    for (a, b), c in pred['change'].items():
        if c in X.CH:
            change[(ref['mb'].get(a), ref['mf'].get(b))] = X.CH[c]
    return dict(pairs=pairs, change=change, burden=X.BU.get(pred['burden']), valid=True)


def main():
    det, _ = A.det_cached()
    patients = list(RUN.LOCKED)
    full = 'OncoState'
    mats = {full: np.stack([X.evaluate(det[(p, A.ONCO)]['ref'], X.det_pred(det[(p, A.ONCO)])) for p in patients])}
    for key, path in ((A.STATELESS, OUT / 'stateless_raw.jsonl'), (A.STATEFUL, OUT / 'generic_stateful_raw.jsonl')):
        by = {}
        for r in A.load_raw(path):
            by.setdefault((r['patient_id'], r['repeat_id']), {})[r['query_id']] = r
        vec = {p: [] for p in patients}
        for (p, rep), qs in by.items():
            if p in vec:
                _, pred = RUN.agent_patient_ev(det[(p, A.ONCO)]['ref'], qs)
                vec[p].append(X.evaluate(det[(p, A.ONCO)]['ref'], to_ref(det[(p, A.ONCO)]['ref'], pred)))
        assert all(vec[p] for p in patients), 'missing agent runs'
        mats[NAMES[key]] = np.stack([np.mean(vec[p], 0) for p in patients])
    avail = {m: np.ones(len(patients), bool) for m in mats}
    point, boot, counts = X.bootstrap(mats, avail, patients)
    rows = [dict(method=m, metric=k, estimate=point[m][k], ci95_lo=np.nanpercentile(boot[m][k], 2.5), ci95_hi=np.nanpercentile(boot[m][k], 97.5),
                 n_evaluated=len(patients)) for m in mats for k in X.METRICS]
    met = pd.DataFrame(rows)
    arch = pd.read_csv(X.BASE / 'external_metrics.csv')
    arch = arch[arch.method == 'Full OncoAgent'].set_index('metric')
    mine = met[met.method == full].set_index('metric')
    diff = max(abs(mine.at[k, c] - arch.at[k, c]) for k in X.METRICS for c in ('estimate', 'ci95_lo', 'ci95_hi') if np.isfinite(arch.at[k, c]))
    print(f'Full OncoState vs archived external_metrics.csv: max |diff| = {diff:.2e}')
    if diff > 1e-9:
        sys.exit('STOP: Full OncoState does not reproduce the archived external metrics')
    met.to_csv(OUT / 'external_metrics_same_definition.csv', index=False)
    pairs = []
    for m in NAMES.values():
        pairs += X.paired_common(mats, avail, full, m, counts)
    pairs += X.paired_common(mats, avail, NAMES[A.STATEFUL], NAMES[A.STATELESS], counts)
    pd.DataFrame(pairs).to_csv(OUT / 'external_paired_same_definition.csv', index=False)
    print(met[met.metric.isin(['match_f1', 'change_accuracy', 'burden_accuracy'])].pivot(index='metric', columns='method', values='estimate').round(4))
    pp = pd.DataFrame(pairs); print(pp[pp.metric.isin(['match_f1', 'change_accuracy', 'burden_accuracy'])].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
