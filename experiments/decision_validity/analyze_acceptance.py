#!/usr/bin/env python3
"""Part A endpoints (analysis plan, section 2) from results/acceptance_per_patient.csv -> tables/A_*.csv, results/acceptance_summary.json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv.common import CFG, exp1_states  # noqa: E402
from dv.stats import boot_idx, cp, holm, paired, topk_error  # noqa: E402

T = HERE / 'tables'; T.mkdir(exist_ok=True)
AL = CFG['acceptance']['alpha_primary']; B, SEED = CFG['bootstrap']['n'], CFG['bootstrap']['seed']; NI = CFG['noninferiority_margin_pp'] / 100


def main():
    D = pd.read_csv(HERE / 'results/acceptance_per_patient.csv', dtype={'patient_id': str})
    D['acc'] = D.action.eq('ACCEPT')
    pids = sorted(D.patient_id.unique()); N = len(pids)
    st1 = exp1_states(); st1 = st1[np.isclose(st1.alpha, AL)].set_index('patient_id').loc[pids]
    IDX = boot_idx(N, B, SEED)

    def arm(method, a=AL):
        d = D[(D.method == method) & np.isclose(D.alpha, a)].set_index('patient_id').loc[pids]
        return d

    def vec(d):
        return d.acc.to_numpy(), d.wrong.to_numpy(float), d.ref_included.to_numpy(bool)

    # ---------------------------------------------------------------- A1: action table, all methods at the primary alpha
    rows = []
    methods = [m for m in D.method.unique()]
    for m in methods:
        for a in sorted(D[D.method == m].alpha.unique()):
            d = arm(m, a); acc, w, inc = vec(d)
            k, n = int(w[acc].sum()), int(acc.sum())
            rows.append(dict(method=m, alpha=a, n=N, accept=n, verify=int(d.action.eq('VERIFY').sum()), abstain=int(d.action.eq('ABSTAIN').sum()),
                             abstain_calibration=int(d.status.eq('ABSTAIN_INSUFFICIENT_CALIBRATION').sum()),
                             timeouts=int(d.status.eq('INCONCLUSIVE_TIMEOUT').sum()), accepted_wrong=k, accepted_correct=n - k,
                             cert_error=k / n if n else np.nan, cert_error_ci=cp(k, n), incorrect_cert_rate=k / N, correct_cert_rate=(n - k) / N,
                             cov_all=inc.mean(), cov_accepted=inc[acc].mean() if n else np.nan,
                             false_pd_accepted=int(d.false_pd.to_numpy()[acc].sum()), missed_pd_accepted=int(d.missed_pd.to_numpy()[acc].sum()),
                             wrong_accepted_ref_excluded=int((w[acc] == 1)[~inc[acc]].sum()), wrong_accepted_ref_included=int((w[acc] == 1)[inc[acc]].sum())))
    A1 = pd.DataFrame(rows); A1.to_csv(T / 'A1_actions_all_methods.csv', index=False)

    # ---------------------------------------------------------------- A2: per-stratum coverage and acceptance, C0 vs CA
    rows = []
    for m in ('C0', 'CA'):
        d = arm(m)
        for s, g in d.groupby('stratum'):
            acc, w, inc = g.acc.to_numpy(), g.wrong.to_numpy(float), g.ref_included.to_numpy(bool)
            rows.append(dict(method=m, stratum=s, n=len(g), cov_all=inc.mean(), cov_all_ci=cp(int(inc.sum()), len(g)), accept=int(acc.sum()),
                             accepted_wrong=int(w[acc].sum()), cov_accepted=inc[acc].mean() if acc.any() else np.nan,
                             median_q=float(g.q.replace(np.inf, np.nan).median()), n_pool_median=float(g.n_pool.median()),
                             abstain=int(g.action.eq('ABSTAIN').sum()), verify=int(g.action.eq('VERIFY').sum())))
    pd.DataFrame(rows).to_csv(T / 'A2_strata_coverage.csv', index=False)

    # ---------------------------------------------------------------- P1 / P2 (and the same contrast at the sensitivity alphas / strata)
    def p1p2(mA, aA, mB, aB):
        cA, wA, _ = vec(arm(mA, aA)); cB, wB, _ = vec(arm(mB, aB))
        f1 = lambda ix: (cA[ix] & (wA[ix] == 1)).mean() - (cB[ix] & (wB[ix] == 1)).mean()   # noqa: E731
        f2 = lambda ix: (cA[ix] & (wA[ix] == 0)).mean() - (cB[ix] & (wB[ix] == 0)).mean()   # noqa: E731
        f3 = lambda ix: (wA[ix][cA[ix]].mean() if cA[ix].any() else np.nan) - (wB[ix][cB[ix]].mean() if cB[ix].any() else np.nan)  # noqa: E731
        o = {}
        for name, f in (('incorrect_cert_rate', f1), ('correct_cert_rate', f2), ('cert_error_among_accepted', f3)):
            pt, ci, p = paired(f, IDX); o[name] = dict(diff=pt, lo=ci[0], hi=ci[1], p=p)
        o['P1_success'] = o['incorrect_cert_rate']['hi'] < 0
        o['P2_success'] = o['correct_cert_rate']['lo'] > -NI
        return o

    prim = p1p2('CA', AL, 'C0', AL)
    sens = {}
    for a in CFG['acceptance']['alpha_sensitivity']:
        sens[f'alpha_{a}'] = p1p2('CA', a, 'C0', a)
    for m in [m for m in methods if m.startswith('CA_')]:
        sens[m] = p1p2(m, AL, 'C0', AL)
    reps = sorted({int(m.split('_r')[-1]) for m in methods if m.startswith('C6_')})
    for r in reps:
        sens[f'hybrid_CA_vs_hybrid_r{r}'] = p1p2(f'C7_hybrid_CA_r{r}', AL, f'C6_hybrid_r{r}', AL)
        sens[f'hybrid_CA_vs_C0_r{r}'] = p1p2(f'C7_hybrid_CA_r{r}', AL, 'C0', AL)
        sens[f'CA_vs_hybrid_r{r}'] = p1p2('CA', AL, f'C6_hybrid_r{r}', AL)
    flat = []
    for k, o in [('primary CA vs C0', prim)] + list(sens.items()):
        for e in ('incorrect_cert_rate', 'correct_cert_rate', 'cert_error_among_accepted'):
            flat.append(dict(contrast=k, endpoint=e, **o[e]))
    pd.DataFrame(flat).to_csv(T / 'A3_P1_P2_and_sensitivity.csv', index=False)

    # ---------------------------------------------------------------- S1: matched acceptance k = |ACCEPT_CA|
    dCA = arm('CA'); cCA, wCA, _ = vec(dCA); k = int(cCA.sum())
    c0_counts = A1[A1.method == 'C0'].set_index('alpha').accept
    grid = CFG['acceptance']['expanded_alpha_grid']
    a_prime = min(grid, key=lambda a: (abs(c0_counts.loc[a] - k), a))
    cC1, wC1, _ = vec(arm('C0', a_prime))
    conf = st1.v0_min_conf.to_numpy(float); gs = st1.gs_agreement.to_numpy(float)
    resid = dCA.residual_mm.to_numpy(float); resid_rank = -np.where(np.isfinite(resid), resid, -1.0)   # missing residual = single = best
    cC0, wC0, _ = vec(arm('C0'))
    w = wCA  # rule-derived wrongness of the V0 decision; identical across V0-energy methods (same argmin)
    assert np.array_equal(w, wC0)

    def s1(ix):
        kb = int(cCA[ix].sum())
        if kb == 0:
            return dict(CA=np.nan, C1=np.nan, C2=np.nan, C3=np.nan, C4=np.nan)
        e_ca = w[ix][cCA[ix]].mean()
        e_c1 = w[ix][cC1[ix]].mean() if cC1[ix].any() else np.nan
        e_c2 = topk_error(conf[ix], w[ix], kb); e_c3 = topk_error(gs[ix], w[ix], kb)
        c0 = cC0[ix]; k4 = min(kb, int(c0.sum()))
        e_c4 = topk_error(resid_rank[ix][c0], w[ix][c0], k4) if k4 else np.nan
        return dict(CA=e_ca, C1=e_c1, C2=e_c2, C3=e_c3, C4=e_c4)

    pt = s1(np.arange(N)); bs = pd.DataFrame([s1(ix) for ix in IDX])
    S1 = []
    for c in ('C1', 'C2', 'C3', 'C4'):
        dd = (bs.CA - bs[c]).dropna().to_numpy()
        p = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
        S1.append(dict(control=c, CA_error=pt['CA'], control_error=pt[c], diff=pt['CA'] - pt[c], lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), p=p))
    S1 = pd.DataFrame(S1); S1['p_holm'] = holm(S1.p.to_numpy())
    S1['k'] = k; S1['alpha_prime_C1'] = a_prime; S1['C1_accept'] = int(cC1.sum()); S1['C4_k'] = min(k, int(cC0.sum()))
    S1.to_csv(T / 'A4_S1_matched_acceptance.csv', index=False)

    # ---------------------------------------------------------------- risk-coverage curves
    rc = []
    for m in ('C0', 'CA'):
        for a in sorted(D[D.method == m].alpha.unique()):
            c, ww, _ = vec(arm(m, a))
            rc.append(dict(method=m, alpha=a, accept=int(c.sum()), coverage=c.mean(), error=ww[c].mean() if c.any() else np.nan,
                           wrong=int(ww[c].sum())))
    for name, sc_ in (('C2_v0_confidence', conf), ('C3_gs_agreement', gs), ('C5_residual', resid_rank)):
        for kk in range(1, N + 1):
            rc.append(dict(method=name, alpha=np.nan, accept=kk, coverage=kk / N, error=topk_error(sc_, w, kk), wrong=np.nan))
    for r in reps:
        c, ww, _ = vec(arm(f'C6_hybrid_r{r}')); rc.append(dict(method=f'C6_hybrid_r{r}', alpha=AL, accept=int(c.sum()), coverage=c.mean(), error=ww[c].mean(), wrong=int(ww[c].sum())))
        c, ww, _ = vec(arm(f'C7_hybrid_CA_r{r}')); rc.append(dict(method=f'C7_hybrid_CA_r{r}', alpha=AL, accept=int(c.sum()), coverage=c.mean(), error=ww[c].mean(), wrong=int(ww[c].sum())))
    pd.DataFrame(rc).to_csv(T / 'A5_risk_coverage.csv', index=False)

    # ---------------------------------------------------------------- fate of the 9 incorrect C0 certificates; CA-specific errors
    d0 = arm('C0'); nine = d0[d0.acc & (d0.wrong == 1)].index
    fate = dCA.loc[nine, ['stratum', 'residual_mm', 'q', 'n_pool', 'status', 'action', 'response', 'ref_response', 'ref_included']].reset_index()
    fate.to_csv(T / 'A6_fate_of_nine.csv', index=False)
    gained = dCA[dCA.acc & ~d0.acc]; lost = d0[d0.acc & ~dCA.acc]
    pd.concat([gained.assign(change='accepted by CA only'), lost.assign(change='accepted by C0 only')])[
        ['change', 'stratum', 'residual_mm', 'status', 'response', 'ref_response', 'wrong', 'ref_included']].reset_index().to_csv(T / 'A7_changed_acceptance.csv', index=False)

    # ---------------------------------------------------------------- A8: hybrid C6 reproduces the saved exp1_hybrid_cdtr statuses
    H = pd.read_csv(HERE.parents[1] / 'experiments/exp1_hybrid_cdtr/results/per_patient_hybrid.csv', dtype={'patient_id': str})
    pd.DataFrame([dict(repeat=r, n=N, status_mismatch=int((arm(f'C6_hybrid_r{r}').status !=
                                                           H[(H.repeat == r) & np.isclose(H.alpha, AL)].set_index('patient_id').status.loc[pids]).sum()))
                  for r in reps]).to_csv(T / 'A8_hybrid_reproduction.csv', index=False)
    # ---------------------------------------------------------------- A9 (post hoc, not pre-specified): CDT-R with all degraded-stratum certificates withheld
    g = d0[d0.acc & (d0.stratum != 'degraded')]
    pd.DataFrame([dict(method='C0 + abstain if residual > 30 mm (post hoc, not pre-specified)', accepted=len(g), incorrect=int(g.wrong.sum()),
                       correct=len(g) - int(g.wrong.sum()))]).to_csv(T / 'A9_posthoc_gate_only.csv', index=False)

    summ = dict(n=N, k_CA=k, alpha_prime=a_prime, primary=prim, S1=S1.to_dict('records'),
                nine_fate=fate.action.value_counts().to_dict(), gained=dict(n=len(gained), wrong=int(gained.wrong.sum())),
                lost=dict(n=len(lost), wrong=int(lost.wrong.sum())),
                table=A1[np.isclose(A1.alpha, AL)].set_index('method')[['accept', 'verify', 'abstain', 'accepted_wrong', 'cert_error', 'cov_accepted']].to_dict('index'))
    (HERE / 'results/acceptance_summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(A1[np.isclose(A1.alpha, AL)][['method', 'accept', 'verify', 'abstain', 'abstain_calibration', 'accepted_wrong', 'cert_error', 'cov_all', 'cov_accepted']].round(3).to_string())
    print(json.dumps(prim, indent=1, default=float)); print(S1.round(4).to_string())
    print(fate.to_string()); print('gained', len(gained), int(gained.wrong.sum()), 'lost', len(lost), int(lost.wrong.sum()))
    print(pd.read_csv(T / 'A2_strata_coverage.csv').round(3).to_string())


if __name__ == '__main__':
    main()
