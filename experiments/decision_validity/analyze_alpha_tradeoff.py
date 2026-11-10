#!/usr/bin/env python3
"""Endpoints of the alpha trade-off and stratification ablation (post hoc analysis-plan amendment) from results/alpha_tradeoff_per_patient.csv.gz.
    python analyze_alpha_tradeoff.py   -> tables/M_*.csv, results/alpha_tradeoff_summary.json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv.common import CFG, quantile  # noqa: E402
from dv.stats import boot_idx  # noqa: E402
from run_alpha_tradeoff import G  # noqa: E402

T = HERE / 'tables'
B, SEED = 2000, CFG['bootstrap']['seed']


def main():
    D = pd.read_csv(HERE / 'results/alpha_tradeoff_per_patient.csv.gz', dtype={'patient_id': str})
    D['acc'] = D.action.eq('ACCEPT')
    pids = sorted(D.patient_id.unique()); N = len(pids); pidx = {p: i for i, p in enumerate(pids)}
    methods = sorted(D.method.unique())
    # arrays: method -> (alpha list, acc[n_alpha, N], wrong[n_alpha, N])
    ARR = {}
    for m in methods:
        d = D[D.method == m]
        al = sorted(d.alpha.unique())
        acc = np.zeros((len(al), N), bool); wr = np.zeros((len(al), N), bool)
        for k, a in enumerate(al):
            x = d[np.isclose(d.alpha, a)]
            ii = x.patient_id.map(pidx).to_numpy()
            acc[k, ii] = x.acc.to_numpy(); wr[k, ii] = (x.wrong.to_numpy() == 1)
        ARR[m] = (al, acc, wr)
    idx = boot_idx(N, B, SEED)

    def curve(m, ix=None):
        al, acc, wr = ARR[m]
        a_, w_ = (acc, wr) if ix is None else (acc[:, ix], wr[:, ix])
        cor = (a_ & ~w_).sum(1).astype(float); inc = (a_ & w_).sum(1).astype(float)
        o = np.argsort(cor, kind='stable')
        c, i_ = cor[o], inc[o]
        uc = np.unique(c)
        return uc, np.array([i_[c == u].mean() for u in uc])

    def dauc(A, Bm, ix=None):
        ca, ia = curve(A, ix); cb, ib = curve(Bm, ix)
        lo, hi = max(ca.min(), cb.min()), min(ca.max(), cb.max())
        if hi <= lo:
            return np.nan
        x = np.linspace(lo, hi, 200)
        return float(np.mean(np.interp(x, ca, ia) - np.interp(x, cb, ib)))

    pairs = [('CA', 'C0'), ('CA_conf', 'C0'), ('CA_nles', 'C0')] + [(f'C7_r{r}', f'C6_r{r}') for r in range(5)] + \
            [(f'C6_r{r}', 'C0') for r in range(5)] + [(f'C7_r{r}', 'CA') for r in range(5)]
    rows = []
    for A, Bm in pairs:
        pt = dauc(A, Bm); bs = np.array([dauc(A, Bm, ix) for ix in idx]); bs = bs[np.isfinite(bs)]
        rows.append(dict(A=A, B=Bm, dAUC_incorrect_per_correct=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         p=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
    TA = pd.DataFrame(rows); TA.to_csv(T / 'M1_tradeoff_dAUC.csv', index=False)

    # curves table
    cr = []
    for m in [x for x in methods if not x.startswith('CA_rand')]:
        al, acc, wr = ARR[m]
        for k, a in enumerate(al):
            cr.append(dict(method=m, alpha=a, accepted=int(acc[k].sum()), incorrect=int((acc[k] & wr[k]).sum()), correct=int((acc[k] & ~wr[k]).sum())))
    CR = pd.DataFrame(cr); CR.to_csv(T / 'M2_tradeoff_points.csv', index=False)

    # matched acceptance over the grid
    mm = []
    for A, Bm in pairs:
        ca, cb = CR[CR.method == A].set_index('alpha'), CR[CR.method == Bm].set_index('alpha')
        for a in ca.index:
            b = min(cb.index, key=lambda x: (abs(cb.loc[x, 'accepted'] - ca.loc[a, 'accepted']), x))
            mm.append(dict(A=A, B=Bm, alpha_A=a, alpha_B=b, acc_A=ca.loc[a, 'accepted'], acc_B=cb.loc[b, 'accepted'],
                           inc_A=ca.loc[a, 'incorrect'], inc_B=cb.loc[b, 'incorrect'],
                           err_A=ca.loc[a, 'incorrect'] / max(1, ca.loc[a, 'accepted']), err_B=cb.loc[b, 'incorrect'] / max(1, cb.loc[b, 'accepted'])))
    MM = pd.DataFrame(mm); MM['diff'] = MM.err_A - MM.err_B; MM.to_csv(T / 'M3_matched_acceptance_grid.csv', index=False)
    msum = MM.groupby(['A', 'B']).agg(n_alpha=('diff', 'size'), A_better=('diff', lambda s: int((s < 0).sum())), tie=('diff', lambda s: int((s == 0).sum())),
                                      B_better=('diff', lambda s: int((s > 0).sum())), mean_diff=('diff', 'mean')).reset_index()
    # bootstrap at alpha_A = 0.10 with alpha_B fixed from the full-sample match
    ci = []
    for _, r in MM[np.isclose(MM.alpha_A, .10)].iterrows():
        alA, accA, wrA = ARR[r.A]; alB, accB, wrB = ARR[r.B]
        kA, kB = alA.index(r.alpha_A) if r.alpha_A in alA else int(np.argmin(np.abs(np.array(alA) - r.alpha_A))), \
            alB.index(r.alpha_B) if r.alpha_B in alB else int(np.argmin(np.abs(np.array(alB) - r.alpha_B)))

        def f(ix):
            a_, w_ = accA[kA, ix], wrA[kA, ix]; b_, v_ = accB[kB, ix], wrB[kB, ix]
            return (w_[a_].mean() if a_.any() else np.nan) - (v_[b_].mean() if b_.any() else np.nan)
        bs = np.array([f(ix) for ix in idx]); bs = bs[np.isfinite(bs)]
        ci.append(dict(A=r.A, B=r.B, diff_at_010=r['diff'], lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    msum = msum.merge(pd.DataFrame(ci), on=['A', 'B']); msum.to_csv(T / 'M3_matched_acceptance_summary.csv', index=False)

    # random strata (alpha 0.10)
    rnd = []
    for m in [x for x in methods if x.startswith('CA_rand')]:
        _, acc, wr = ARR[m]
        rnd.append(dict(method=m, accepted=int(acc[0].sum()), incorrect=int((acc[0] & wr[0]).sum()), correct=int((acc[0] & ~wr[0]).sum())))
    RND = pd.DataFrame(rnd); RND.to_csv(T / 'M4_random_strata.csv', index=False)
    ref = CR[(CR.method.isin(['CA', 'CA_conf', 'CA_nles', 'C0'])) & np.isclose(CR.alpha, .10)].set_index('method')
    rsum = dict(n=len(RND), incorrect_mean=RND.incorrect.mean(), incorrect_min=int(RND.incorrect.min()), incorrect_q05=float(RND.incorrect.quantile(.05)),
                correct_mean=RND.correct.mean(), CA_incorrect=int(ref.loc['CA', 'incorrect']), CA_correct=int(ref.loc['CA', 'correct']),
                share_random_le_CA_incorrect=float((RND.incorrect <= ref.loc['CA', 'incorrect']).mean()),
                share_random_le_CA_incorrect_and_ge_correct=float(((RND.incorrect <= ref.loc['CA', 'incorrect']) & (RND.correct >= ref.loc['CA', 'correct'])).mean()),
                ablation_010=ref[['accepted', 'incorrect', 'correct']].to_dict('index'))

    # coverage-gap diagnostic from calibration pools only
    PV = pd.read_csv(HERE / 'results/alpha_tradeoff_patient_vars.csv', dtype={'patient_id': str})
    pv = PV.set_index('patient_id'); coh = pv[pv.in_cohort]; ext = pv[~pv.in_cohort]
    gaps = []
    for col, use_ext in [('v0_score', True)] + [(f'hyb_score_r{r}', False) for r in range(5)]:
        g_ = []
        for p, rp in coh.iterrows():
            pool = coh[coh.fold != rp.fold]
            if use_ext:
                pool = pd.concat([pool, ext])
            q = quantile(pool[col].to_numpy(), .10)
            pl = pool[pool.stratum != 'single']
            cd = (pl[pl.stratum == 'degraded'][col] <= q + 1e-9).mean(); cr_ = (pl[pl.stratum == 'reliable'][col] <= q + 1e-9).mean()
            g_.append((cd, cr_))
        g_ = np.array(g_)
        gaps.append(dict(energy=col, cov_degraded=g_[:, 0].mean(), cov_reliable=g_[:, 1].mean(), gap=(g_[:, 0] - g_[:, 1]).mean()))
    GAP = pd.DataFrame(gaps); GAP.to_csv(T / 'M5_calibration_coverage_gap.csv', index=False)

    # degraded-stratum subgroup at alpha 0.10
    sub = []
    for m in ['C0', 'CA', 'CA_conf', 'CA_nles'] + [f'C6_r{r}' for r in range(5)] + [f'C7_r{r}' for r in range(5)]:
        d = D[(D.method == m) & np.isclose(D.alpha, .10)]
        for s in ('degraded', 'reliable'):
            x = d[d.stratum == s]
            sub.append(dict(method=m, stratum=s, n=len(x), accepted=int(x.acc.sum()), incorrect=int((x.acc & (x.wrong == 1)).sum()),
                            coverage=x.ref_included.mean()))
    SUB = pd.DataFrame(sub); SUB.to_csv(T / 'M6_strata_subgroup.csv', index=False)

    summ = dict(tradeoff=TA.to_dict('records'), matched=msum.to_dict('records'), random=rsum,
                coverage_gap=GAP.to_dict('records'))
    (HERE / 'results/alpha_tradeoff_summary.json').write_text(json.dumps(summ, indent=1, default=float))
    pd.set_option('display.width', 250)
    print(TA.round(3).to_string()); print(msum.round(3).to_string()); print(json.dumps(rsum, indent=1, default=float))
    print(GAP.round(3).to_string()); print(SUB.round(3).to_string())
    print(CR[CR.method.isin(['C0', 'CA', 'C6_r0', 'C7_r0'])].pivot_table(index='alpha', columns='method', values=['correct', 'incorrect']).to_string())


if __name__ == '__main__':
    main()
