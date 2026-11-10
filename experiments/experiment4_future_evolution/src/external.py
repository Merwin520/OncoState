#!/usr/bin/env python3
"""4E I-SPY1 external: LD-only next-visit forecasting. Model trained on all I-SPY2 patients with LD (frozen), applied unchanged to I-SPY1
(LD, HR, HER2, age; study days unavailable in I-SPY1, so no horizon feature). Internal I-SPY2 patient-level 5-fold CV for reference.
Writes results/4E_ispy1_ld.csv.  python external.py"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import forecast as FC
from check import check
from common import ISPY1, ISPY2, NB, RES, SEED, T2, F

NEXT = FC.NEXT


def feat(tab, a, clin):
    x = F(tab, FC.IDX[a], clin, names=('ld',)); x.pop('day_latest', None); return x


def main():
    check()
    I = ISPY2(); rows2 = {}
    for p in I.pts:
        st = I.states(p, 'natural'); c = I.clinical(p); clin = dict(HR=c['HR'], HER2=c['HER2'], age=c['age'])
        tabs = [I.table_from_state(st[k], k) for k in range(4)]
        rows2[p] = (tabs, clin)
    E = ISPY1(); rows1 = {}
    for p in E.pts:
        st = E.states(p); c = E.clinical(p)
        rows1[p] = ([E.table_from_state(st[k], k) for k in range(4)], dict(HR=c['HR'], HER2=c['HER2'], age=c['age']))
    out = []
    for a, b in NEXT:
        def build(rows):
            X, yv, ids = [], [], []
            for p, (tabs, clin) in rows.items():
                ta, tb = tabs[FC.IDX[a]].get(a), tabs[3].get(b)
                if ta is None or tb is None or ta.get('ld') is None or tb.get('ld') is None:
                    continue
                X.append(feat(tabs[FC.IDX[a]], a, clin)); yv.append(np.log1p(tb['ld'])); ids.append(p)
            return pd.DataFrame(X, index=ids), np.array(yv)
        X2, y2 = build(rows2); X1, y1 = build(rows1); X1 = X1.reindex(columns=X2.columns)
        last2, last1 = np.log1p(np.exp(X2[f'{a}_ld']) - .1), np.log1p(np.exp(X1[f'{a}_ld']) - .1)
        for kind in ('ridge', 'gbm', 'mean', 'locf'):
            if kind == 'locf':
                p1, pin = last1.to_numpy(), last2.to_numpy()
            elif kind == 'mean':
                p1 = np.full(len(y1), y2.mean()); pin = np.zeros(len(y2))
                for tr, te in KFold(5, shuffle=True, random_state=0).split(X2):
                    pin[te] = y2[tr].mean()
            else:
                p1 = FC.model(kind).fit(X2.to_numpy(float), y2).predict(X1.to_numpy(float)); pin = np.zeros(len(y2))
                for tr, te in KFold(5, shuffle=True, random_state=0).split(X2):
                    pin[te] = FC.model(kind).fit(X2.iloc[tr].to_numpy(float), y2[tr]).predict(X2.iloc[te].to_numpy(float))
            for cohort, yy, pp in (('I-SPY1 external', y1, p1), ('I-SPY2 internal CV', y2, pin)):
                e = np.abs(pp - yy); rng = np.random.default_rng(SEED); b_ = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(NB)]
                out.append(dict(transition=f'{a}->{b}', cohort=cohort, model=kind, n=len(yy), MAE_log=e.mean(), lo=np.percentile(b_, 2.5),
                                hi=np.percentile(b_, 97.5), MAE_cm=np.mean(np.abs(np.expm1(pp) - np.expm1(yy)))))
    O = pd.DataFrame(out); O.to_csv(RES / '4E_ispy1_ld.csv', index=False)
    pd.set_option('display.width', 200); print(O.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
