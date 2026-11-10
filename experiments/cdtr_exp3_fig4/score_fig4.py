#!/usr/bin/env python3
"""Score V0-energy arms (experiments/cdtr runs) and gated-energy arms on the 144 future patients under the repository engine and the
exp30 response rule; paired patient bootstrap."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[0] / 'cdtr'
sys.path.insert(0, str(P1)); sys.path.insert(0, str(HERE.parents[0] / 'exp33_clinical_propagation'))
from cdtr.reference import ReferenceForSplit  # noqa: E402
from cdtr.rules import response  # noqa: E402
from rules import assess  # noqa: E402

EV = json.loads((HERE / 'results/evidence_gated.json').read_text())['patients']
TEST = sorted(p for p, e in EV.items() if e['split'] == 'test')
ref = ReferenceForSplit('test', final_evaluation=True)


def outcome(p, pairs):
    e = EV[p]; pairs = [tuple(x) for x in pairs]
    r_eng, inp, _ = response(e['bl_t'], e['fu_t'], pairs)
    case = dict(baseline_lesions=[dict(lesion_id=k, diameter_mm=v / 10) for k, v in e['bl_t'].items()],
                followup_lesions=[dict(lesion_id=k, diameter_mm=v / 10) for k, v in e['fu_t'].items()])
    a = assess(case, pairs)
    return r_eng, inp['new_lesions'], a['response'], a['new_lesion']


TRUTH = {p: outcome(p, ref.pairs(p)) for p in TEST}


def rows_for(arm, blob):
    out = []
    for seed, run in blob['runs'].items():
        for K, recs in run['checkpoints'].items():
            for p, r in recs.items():
                re_, ne, ra, na = outcome(p, r['pairs']); te, tne, ta, tna = TRUTH[p]
                for rule, y, yt, n, nt in (('engine', re_, te, ne, tne), ('exp30_rule', ra, ta, na, tna)):
                    out.append(dict(arm=arm, seed=int(seed), K=int(K), patient_id=p, rule=rule, wrong=int(y != yt),
                                    false_pd=int(y == 'PD' and yt != 'PD'), missed_pd=int(y != 'PD' and yt == 'PD'),
                                    incorrect_pd=int((y == 'PD') != (yt == 'PD')), false_new=int(n and not nt), missed_new=int(nt and not n),
                                    n_reviews=r['n_reviews']))
    return out


rows = []
for f in sorted((P1 / 'results/test').glob('*.json')):
    if 'alpha' in f.stem:
        continue
    rows += rows_for(f'V0+{f.stem}', json.loads(f.read_text()))
for f in sorted((HERE / 'results/test').glob('*.json')):
    rows += rows_for(f.stem.replace('gated_', 'gated+'), json.loads(f.read_text()))
d = pd.DataFrame(rows)
d.to_csv(HERE / 'tables/patient_level.csv', index=False)
E = ['wrong', 'false_pd', 'missed_pd', 'incorrect_pd', 'false_new', 'missed_new']
per = d.groupby(['rule', 'arm', 'K', 'patient_id'])[E].mean().reset_index()
summ = per.groupby(['rule', 'arm', 'K'])[E].mean().reset_index()
summ.to_csv(HERE / 'tables/summary.csv', index=False)
rng = np.random.default_rng(20261011); idx = rng.integers(0, len(TEST), (2000, len(TEST)))
pd_rows = []
for rule in ('engine', 'exp30_rule'):
    P = per[per.rule == rule].set_index(['arm', 'K', 'patient_id'])
    def vec(arm, K, e):
        return P.loc[(arm, K)].loc[TEST][e].to_numpy()
    comps = [('gated+cdtr', 'gated+no_review'), ('gated+cdtr', 'V0+cdtr'), ('gated+cdtr', 'gated+model_confidence'),
             ('gated+cdtr', 'gated+set_ambiguity'), ('gated+cdtr', 'gated+random'), ('V0+cdtr', 'gated+no_review'),
             ('gated+no_review', 'V0+v0_no_review')]
    for a, b in comps:
        for K in sorted(per.K.unique()):
            if (a, K) not in P.index.droplevel(2) or (b, K) not in P.index.droplevel(2):
                continue
            Kb = 0 if b.endswith('no_review') else K
            for e in ('wrong', 'incorrect_pd', 'missed_new'):
                dl = vec(a, K, e) - vec(b, Kb, e); bs = dl[idx].mean(1)
                pd_rows.append(dict(rule=rule, comparison=f'{a} - {b}', K=K, endpoint=e, diff=dl.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
pdf = pd.DataFrame(pd_rows); pdf.to_csv(HERE / 'tables/paired_differences.csv', index=False)
pd.set_option('display.width', 220)
for rule in ('engine', 'exp30_rule'):
    s = summ[(summ.rule == rule) & summ.K.isin([0, 36, 72, 144, 288])]
    print(rule); print((s.pivot_table(index='arm', columns='K', values='wrong') * 100).round(1).to_string()); print()
x = pdf[pdf.K.isin([0, 72, 144, 288]) & (pdf.endpoint == 'wrong')]
print(x.round(3).to_string(index=False))
