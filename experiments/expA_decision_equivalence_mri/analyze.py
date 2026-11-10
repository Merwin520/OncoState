#!/usr/bin/env python3
"""Experiment A endpoints (pre-specified analysis plan): acceptance, false-certification risk, coverage, matched-acceptance comparisons
(paired patient bootstrap, Holm), risk-coverage curves, review budget (A3), revision (A4), lambda sensitivity.
    python analyze.py   -> tables/*.csv, results/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'decision_validity'))
from dv.stats import cp, holm, topk_error  # noqa: E402

RES, TAB = HERE / 'results', HERE / 'tables'
B, SEED = 10_000, 20261026
ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def acc_row(df, mask, name, **kw):
    a = df[mask]; n, k = len(a), int(a.wrong.sum())
    lo, hi = cp(k, n)
    return dict(method=name, accepted=n, accepted_frac=n / len(df), patients=a.patient.nunique(), wrong=k, risk=k / n if n else np.nan,
                risk_lo=lo, risk_hi=hi, false_pd=int(((a.point == 'PD') & (a.expert != 'PD')).sum()),
                missed_pd=int(((a.point != 'PD') & (a.expert == 'PD')).sum()), **kw)


def main():
    TAB.mkdir(exist_ok=True)
    df = pd.read_csv(RES / 'decisions.csv')
    N = len(df); pts = sorted(df.patient.unique())
    out = {'n_decisions': N, 'n_patients': len(pts), 'expert': df.expert.value_counts().to_dict(), 'point': df.point.value_counts().to_dict(),
           'point_accuracy': float(1 - df.wrong.mean()), 'confusion': pd.crosstab(df.expert, df.point).to_dict()}
    rows = [acc_row(df, np.ones(N, bool), 'C1 point estimate (accept all)')]
    for al in ALPHAS:
        rows.append(acc_row(df, df[f'cert_{al}'] == 1, 'CDT-R', alpha=al,
                            coverage_patients=float(df.drop_duplicates('patient')[f'cover_{al}'].mean()),
                            coverage_certified=float(df[df[f'cert_{al}'] == 1][f'cover_{al}'].mean()) if (df[f'cert_{al}'] == 1).any() else np.nan,
                            q_median=float(df[f'q_{al}'].median())))
        rows.append(acc_row(df, df[f'certf_{al}'] == 1, 'C3 current-scan interval', alpha=al, q_median=float(df[f'qf_{al}'].median())))
    rows.append(acc_row(df, df.db_point == df.point, 'C4 two-pipeline agreement'))
    A = pd.DataFrame(rows); A.to_csv(TAB / 'A2_acceptance_risk.csv', index=False)

    # ---------------------------------------------------------------- matched-acceptance comparisons (primary)
    pid = df.patient.to_numpy(); P = np.array(pts); idx_of = {p: np.where(pid == p)[0] for p in pts}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([idx_of[p] for p in rng.choice(P, len(P))]) for _ in range(B)]
    w = df.wrong.to_numpy(float); flip = df.flip.to_numpy(float); unc = -df.disagreement.to_numpy(float)
    certA = (df['cert_0.1'] == 1).to_numpy(); certC3 = (df['certf_0.1'] == 1).to_numpy(); agree = (df.db_point == df.point).to_numpy()

    def comp(ix):
        r = {}
        k = int(certA[ix].sum())
        r['1'] = (w[ix][certA[ix]].mean() if k else np.nan) - (topk_error(unc[ix], w[ix], k) if k else np.nan)
        k3 = int(certC3[ix].sum())
        r['2'] = (topk_error(flip[ix], w[ix], k3) if k3 else np.nan) - (w[ix][certC3[ix]].mean() if k3 else np.nan)
        k4 = int(agree[ix].sum())
        r['3'] = (topk_error(flip[ix], w[ix], k4) if k4 else np.nan) - (w[ix][agree[ix]].mean() if k4 else np.nan)
        return r

    pt = comp(np.arange(N)); bs = pd.DataFrame([comp(ix) for ix in draws])
    lab = {'1': ('CDT-R (alpha 0.10)', 'C2 uncertainty abstention', int(certA.sum())),
           '2': ('CDT-R top-k (min flip cost)', 'C3 current-scan interval (alpha 0.10)', int(certC3.sum())),
           '3': ('CDT-R top-k (min flip cost)', 'C4 two-pipeline agreement', int(agree.sum()))}
    S1 = []
    for key, (a_, b_, k) in lab.items():
        v = bs[key].dropna().to_numpy()
        p = float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))) if len(v) else np.nan
        S1.append(dict(comparison=f'{a_} - {b_}', k=k, diff_error=pt[key], lo=float(np.percentile(v, 2.5)) if len(v) else np.nan,
                       hi=float(np.percentile(v, 97.5)) if len(v) else np.nan, p=p))
    S1 = pd.DataFrame(S1); S1['p_holm'] = holm(S1.p.fillna(1).to_numpy()); S1.to_csv(TAB / 'A2_primary_matched.csv', index=False)

    # ---------------------------------------------------------------- risk-coverage curves
    curve = []
    for k in range(1, N + 1):
        curve.append(dict(k=k, coverage=k / N, cdtr=topk_error(flip, w, k), c2=topk_error(unc, w, k)))
    pd.DataFrame(curve).to_csv(TAB / 'A2_risk_coverage.csv', index=False)

    # ---------------------------------------------------------------- A3 review budget
    rev = []
    rr = np.random.default_rng(SEED + 1)
    for K in (0, .05, .1, .2, .3, .5):
        k = int(round(K * N))
        for name, score in (('CDT-R', flip), ('C2 uncertainty', unc)):
            o = np.argsort(score, kind='stable')[:k]                         # least robust / most uncertain first
            ww = w.copy(); ww[o] = 0
            rev.append(dict(budget=K, k=k, order=name, errors=int(ww.sum()), error_rate=ww.mean()))
        rand = np.mean([np.delete(w, rr.choice(N, k, replace=False)).sum() for _ in range(2000)]) if k else w.sum()
        rev.append(dict(budget=K, k=k, order='random', errors=float(rand), error_rate=float(rand / N)))
    R3 = pd.DataFrame(rev); R3.to_csv(TAB / 'A3_review_budget.csv', index=False)
    a3 = []
    for K in (.1, .2):
        k = int(round(K * N))

        def red(ix, score):
            o = np.argsort(score[ix], kind='stable')[:k]; ww = w[ix].copy(); ww[o] = 0; return ww.mean()
        d = np.array([red(ix, flip) - red(ix, unc) for ix in draws[:2000]])
        a3.append(dict(budget=K, diff_cdtr_minus_c2=red(np.arange(N), flip) - red(np.arange(N), unc),
                       lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
    pd.DataFrame(a3).to_csv(TAB / 'A3_paired.csv', index=False)

    # ---------------------------------------------------------------- A4 revision
    Rv = pd.read_csv(RES / 'revision.csv')
    ev = Rv.groupby(['patient', 'revised'])
    a4 = dict(revision_events=int(ev.ngroups), affected_decisions=int(len(Rv)),
              category_changed=int((Rv.old_point != Rv.new_point).sum()),
              certified_before=int(Rv.old_cert.sum()), stale_without_reverification=int(Rv.stale.sum()),
              events_with_stale=int(ev.stale.max().sum()),
              verifier_calls_dependency_aware=int(len(Rv)),
              verifier_calls_full_rebuild=int(ev.n_patient_decisions.first().sum()),
              wrong_before=int((Rv.old_point != Rv.expert).sum()), wrong_after=int((Rv.new_point != Rv.expert).sum()))
    out['A4'] = a4

    # ---------------------------------------------------------------- lambda sensitivity
    sens = []
    for lam in ('0.5', '2'):
        f = RES / f'decisions_lam{lam}.csv'
        if f.exists():
            x = pd.read_csv(f)
            sens.append(acc_row(x, x['cert_0.1'] == 1, 'CDT-R', alpha=.1, lam=float(lam)))
    pd.DataFrame(sens).to_csv(TAB / 'A_sensitivity_lambda.csv', index=False)
    out['primary_alpha_0.1'] = A[(A.method == 'CDT-R') & (A.alpha == 0.1)].iloc[0].to_dict()
    out['S1'] = S1.to_dict('records')
    (RES / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    pd.set_option('display.width', 250)
    print(A.round(3).to_string(index=False)); print(S1.round(4).to_string(index=False)); print(R3.round(3).to_string(index=False))
    print(pd.DataFrame(a3).round(4).to_string(index=False)); print(json.dumps(a4, indent=1)); print(pd.DataFrame(sens).round(3).to_string(index=False))
    print('point accuracy', round(out['point_accuracy'], 3)); print(pd.crosstab(df.expert, df.point))


if __name__ == '__main__':
    main()
