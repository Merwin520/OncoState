#!/usr/bin/env python3
"""Experiment 1 add-on: clinical decision reliability under lesion uncertainty (retrospective secondary analysis).
Reads saved CDT-R Exp 1 records, saved evidence/reference and saved hybrid-matcher predictions; re-runs no model, matcher or verifier.
    python run_analysis.py   -> results/per_patient.csv, results/meta.json, tables/*.csv"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, mannwhitneyu

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P1 = REPO / 'experiments/cdtr'
EXP1 = REPO / 'experiments/cdtr_exp1_state'
HYB = REPO / 'outputs/hybrid_matcher/run/predictions.csv'
sys.path.insert(0, str(P1))
from cdtr.energy import best, prepare  # noqa: E402
from cdtr.rules import rule_inputs  # noqa: E402

B_N, SEED, G = 10_000, 20261019, 30.0
PRIMARY = ('crossfit', 0.10)
COVERAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
RES, TAB = HERE / 'results', HERE / 'tables'
RES.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)


def cp(k, n):
    """Clopper-Pearson exact 95% CI."""
    lo = 0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(.975, k + 1, n - k)
    return lo, hi


def pct(a):
    return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


def topk_error(score, wrong, k):
    """Expected number of errors among the k highest-score patients under uniform random tie-breaking (score: higher = accept first)."""
    if k <= 0:
        return 0.0
    order = np.argsort(-score, kind='stable')
    s = score[order]; w = wrong[order]
    t = s[k - 1]
    above = s > t; tied = s == t
    n_above = above.sum()
    return w[above].sum() + (k - n_above) / tied.sum() * w[tied].sum()


# ------------------------------------------------------------------------------------------------------------------ data
states = pd.read_csv(EXP1 / 'results/per_patient_states.csv', dtype={'patient_id': str})
ev = json.loads((P1 / 'results/evidence.json').read_text())
ref = {p: [tuple(x) for x in v['pairs']] for p, v in json.loads((P1 / 'results/reference_HIDDEN.json').read_text()).items()}
prim = states[(states.variant == PRIMARY[0]) & np.isclose(states.alpha, PRIMARY[1])].set_index('patient_id').sort_index()
assert len(prim) == 274 and prim.index.is_unique
assert states.sound_violation.sum() == 0

# ---- per-patient mechanism descriptors (deterministic, from saved evidence + reference)
rows = []
for p, r in prim.iterrows():
    P = prepare(ev[p], G); x0, F0 = best(P)
    y0, i0, _ = P.response(x0); yr, ir, _ = P.response(ref[p])
    assert y0 == r.v0_response and yr == r.ref_response, p
    assert abs((P.F(ref[p]) - F0) - r.score) < 1e-6, p
    bi, fi = {b: k for k, b in enumerate(P.bids)}, {f: k for k, f in enumerate(P.fids)}
    dref = [P.d[bi[a], fi[b]] for a, b in ref[p]]
    v0_fu = {b for _, b in x0}; ref_fu = {b for _, b in ref[p]}
    spurious_new = sorted(f for f in ref_fu - v0_fu if P.fu_t[f] >= 100)      # measurable FU lesions V0 calls new, reference matches
    driver = ''
    if y0 == 'PD' and yr != 'PD':
        driver = 'new-lesion PD' if (i0['new_lesions'] and not ir['new_lesions']) else 'sum-increase PD'
    elif y0 != yr:
        driver = f'{yr}->{y0}'
    rows.append(dict(patient_id=p, certified=int(r.status == 'PROVED_EQUIVALENT'), status=r.status, wrong=int(r.wrong),
                     false_pd=int(r.false_pd), missed_pd=int(r.missed_pd), v0_response=y0, ref_response=yr,
                     ref_included=int(r.graph_covered), ref_response_reachable=int(r.label_covered), q=r.q, ref_energy_gap=r.score,
                     ref_excess_over_q=r.score - r.q, residual_mm=r.residual, n_bl=P.n, n_fu=P.m, n_ref_pairs=len(ref[p]),
                     ref_pairs_beyond_gate=int(sum(d > G for d in dref)), max_ref_pair_mm=max(dref) if dref else np.nan,
                     n_spurious_new=len(spurious_new), error_driver=driver, v0_min_conf=r.v0_min_conf,
                     ambiguous_lesions=int(r.ambiguous_lesions), gs_agreement=r.gs_agreement, n_reachable=int(r.n_reachable)))
pp = pd.DataFrame(rows).set_index('patient_id')

# ---- hybrid matcher (exploratory): rule-derived response of each saved CV repeat; is its assignment inside C_alpha?
hyb = pd.read_csv(HYB, dtype={'patient_id': str})
hyb = hyb[hyb.matcher == 'hybrid']
reps = sorted(hyb.repeat.unique())
for rep in reps:
    h = hyb[hyb.repeat == rep]
    pairs = {p: list(zip(g.bl, g.fu)) for p, g in h.groupby('patient_id')}
    yy, inside = [], []
    for p in pp.index:
        P = prepare(ev[p], G); _, F0 = best(P)
        x = pairs.get(p, [])
        yy.append(P.response(x)[0]); inside.append(int(P.F(x) - F0 <= prim.loc[p, 'q'] + 1e-9))
    pp[f'hyb{rep}_response'] = yy
    pp[f'hyb{rep}_wrong'] = (pp[f'hyb{rep}_response'] != pp.ref_response).astype(int)
    pp[f'hyb{rep}_in_set'] = inside
pp.to_csv(RES / 'per_patient.csv')

N = len(pp); cert = pp.certified.values.astype(bool); wrong = pp.wrong.values.astype(float); inc = pp.ref_included.values.astype(bool)
rng = np.random.default_rng(SEED)
IDX = rng.integers(0, N, size=(B_N, N))

# ================================================================================================================== A
def a_stats(ix):
    c, w, i = cert[ix], wrong[ix], inc[ix]
    ce = w[c].mean() if c.any() else np.nan; ue = w[~c].mean() if (~c).any() else np.nan
    return dict(cert_rate=c.mean(), cert_err=ce, uncert_err=ue, diff=ce - ue, ratio=ce / ue if ue else np.nan,
                coverage=i.mean(), coverage_cert=i[c].mean(), coverage_uncert=i[~c].mean(),
                err_cert_included=w[c & i].mean() if (c & i).any() else np.nan,
                err_cert_excluded=w[c & ~i].mean() if (c & ~i).any() else np.nan, overall_err=w.mean())


pt = a_stats(np.arange(N))
boot = pd.DataFrame([a_stats(ix) for ix in IDX])
counts = dict(cert_rate=(cert.sum(), N), cert_err=(int(wrong[cert].sum()), cert.sum()), uncert_err=(int(wrong[~cert].sum()), (~cert).sum()),
              coverage=(inc.sum(), N), coverage_cert=(int(inc[cert].sum()), cert.sum()), coverage_uncert=(int(inc[~cert].sum()), (~cert).sum()),
              err_cert_included=(int(wrong[cert & inc].sum()), (cert & inc).sum()), err_cert_excluded=(int(wrong[cert & ~inc].sum()), (cert & ~inc).sum()),
              overall_err=(int(wrong.sum()), N))
ta = []
for k, v in pt.items():
    lo, hi = pct(boot[k].values)
    row = dict(metric=k, estimate=v, boot_lo=lo, boot_hi=hi)
    if k in counts:
        kk, nn = counts[k]; row.update(k=kk, n=nn, exact_lo=cp(kk, nn)[0], exact_hi=cp(kk, nn)[1])
    ta.append(row)
ta = pd.DataFrame(ta); ta.to_csv(TAB / 'A_certification_error.csv', index=False)
ta_p = dict(p_boot_diff_ge0=float((boot['diff'] >= 0).mean()))

# error type breakdown by group
grp = np.where(cert & inc, 'certified, reference included', np.where(cert, 'certified, reference excluded', 'uncertified'))
pp['group'] = grp
tb = pp.groupby('group').agg(n=('wrong', 'size'), wrong=('wrong', 'sum'), false_pd=('false_pd', 'sum'), missed_pd=('missed_pd', 'sum'))
tb.to_csv(TAB / 'A_error_types_by_group.csv')

# sensitivity across all alpha x variant (all reported)
sens = []
for (var, a), g in states.groupby(['variant', 'alpha']):
    c = g.status.eq('PROVED_EQUIVALENT').values; w = g.wrong.values; i = g.graph_covered.values.astype(bool)
    sens.append(dict(variant=var, alpha=a, cert_rate=c.mean(), n_cert=c.sum(), cert_err=w[c].mean() if c.any() else np.nan,
                     n_cert_wrong=int(w[c].sum()), uncert_err=w[~c].mean(), coverage=i.mean(), coverage_cert=i[c].mean() if c.any() else np.nan,
                     cert_wrong_excluded=int(w[c & ~i].sum()), cert_wrong_included=int(w[c & i].sum())))
pd.DataFrame(sens).to_csv(TAB / 'A_sensitivity_alpha_variant.csv', index=False)

# ================================================================================================================== B
SC = {'Matching confidence': pp.v0_min_conf.values.astype(float), 'Matching ambiguity': -pp.ambiguous_lesions.values.astype(float),
      'GS decision agreement': pp.gs_agreement.values.astype(float)}


def b_stats(ix):
    c, w = cert[ix], wrong[ix]; k = int(c.sum())
    out = {'CDT-R': w[c].sum() / k if k else np.nan}
    for name, s in SC.items():
        out[name] = topk_error(s[ix], w, k) / k if k else np.nan
    return out


bp = b_stats(np.arange(N)); bb = pd.DataFrame([b_stats(ix) for ix in IDX])
rows_b = []
for name in bp:
    lo, hi = pct(bb[name].values)
    r = dict(method=name, coverage=cert.mean(), n_accepted=int(cert.sum()), error=bp[name], lo=lo, hi=hi)
    if name != 'CDT-R':
        d = bb['CDT-R'] - bb[name]; r.update(cdtr_minus=bp['CDT-R'] - bp[name], diff_lo=pct(d.values)[0], diff_hi=pct(d.values)[1],
                                            p_boot=float(2 * min((d >= 0).mean(), (d <= 0).mean())))
    rows_b.append(r)
pooled = (bb['CDT-R'] + bb['GS decision agreement']) / 2 - (bb['Matching confidence'] + bb['Matching ambiguity']) / 2
pooled_pt = (bp['CDT-R'] + bp['GS decision agreement']) / 2 - (bp['Matching confidence'] + bp['Matching ambiguity']) / 2
rows_b.append(dict(method='pooled: decision-level minus lesion-level', coverage=cert.mean(), n_accepted=int(cert.sum()), error=pooled_pt,
                   lo=pct(pooled.values)[0], hi=pct(pooled.values)[1], p_boot=float(2 * min((pooled >= 0).mean(), (pooled <= 0).mean()))))
pd.DataFrame(rows_b).to_csv(TAB / 'B_matched_coverage.csv', index=False)

# selective clinical error curve (pre-fixed coverages) + CDT-R alpha points with baselines at matched coverage
curve = []
for cov in COVERAGES:
    k = int(round(cov * N))
    for name, s in SC.items():
        e = topk_error(s, wrong, k) / k
        eb = np.array([topk_error(s[ix], wrong[ix], k) / k for ix in IDX[:2000]])
        curve.append(dict(method=name, coverage=k / N, k=k, error=e, lo=pct(eb)[0], hi=pct(eb)[1]))
for a, g in states[states.variant == 'crossfit'].groupby('alpha'):
    g = g.set_index('patient_id').loc[pp.index]
    c = g.status.eq('PROVED_EQUIVALENT').values; k = int(c.sum())
    curve.append(dict(method='CDT-R', alpha=a, coverage=k / N, k=k, error=wrong[c].sum() / k, lo=cp(int(wrong[c].sum()), k)[0], hi=cp(int(wrong[c].sum()), k)[1]))
    sca = {'Matching confidence': SC['Matching confidence'], 'Matching ambiguity': -g.ambiguous_lesions.values.astype(float),
           'GS decision agreement': SC['GS decision agreement']}
    for name, s in sca.items():
        curve.append(dict(method=f'{name} @CDT-R alpha', alpha=a, coverage=k / N, k=k, error=topk_error(s, wrong, k) / k))
pd.DataFrame(curve).to_csv(TAB / 'B_selective_curve.csv', index=False)

# ================================================================================================================== C
cw = pp[(pp.certified == 1) & (pp.wrong == 1)]; cc = pp[(pp.certified == 1) & (pp.wrong == 0)]
cols = ['residual_mm', 'ref_excess_over_q', 'ref_pairs_beyond_gate', 'max_ref_pair_mm', 'n_spurious_new', 'n_bl', 'n_fu', 'n_reachable']
desc = []
for name, g in [('certified, wrong', cw), ('certified, correct, reference included', cc[cc.ref_included == 1]),
                ('certified, correct, reference excluded', cc[cc.ref_included == 0]), ('uncertified', pp[pp.certified == 0])]:
    r = dict(group=name, n=len(g), residual_missing=int(g.residual_mm.isna().sum()), ref_included=int(g.ref_included.sum()),
             any_ref_pair_beyond_gate=int((g.ref_pairs_beyond_gate > 0).sum()), any_spurious_new=int((g.n_spurious_new > 0).sum()))
    for c in cols:
        r[f'{c}_median'] = g[c].median(); r[f'{c}_q1'] = g[c].quantile(.25); r[f'{c}_q3'] = g[c].quantile(.75)
    desc.append(r)
pd.DataFrame(desc).to_csv(TAB / 'C_group_descriptors.csv', index=False)
keep = ['v0_response', 'ref_response', 'error_driver', 'ref_included', 'ref_response_reachable', 'residual_mm', 'ref_excess_over_q',
        'ref_pairs_beyond_gate', 'max_ref_pair_mm', 'n_spurious_new', 'n_bl', 'n_fu'] + [f'hyb{r}_response' for r in reps] + [f'hyb{r}_in_set' for r in reps]
cw[keep].sort_values('residual_mm').to_csv(TAB / 'C_incorrectly_certified_cases.csv')

x, y = cw.residual_mm.dropna(), cc.residual_mm.dropna()
auc = mannwhitneyu(x, y).statistic / (len(x) * len(y))
ab = []
r2 = np.random.default_rng(SEED + 1)
for _ in range(B_N):
    xs, ys = r2.choice(x.values, len(x)), r2.choice(y.values, len(y))
    ab.append(mannwhitneyu(xs, ys).statistic / (len(xs) * len(ys)))
hy = []
for rep in reps:
    hy.append(dict(repeat=rep, hybrid_overall_error=pp[f'hyb{rep}_wrong'].mean(), v0_overall_error=pp.wrong.mean(),
                   cert_wrong_recovered=int((cw[f'hyb{rep}_wrong'] == 0).sum()), cert_wrong_n=len(cw),
                   cert_wrong_hybrid_in_set=int(cw[f'hyb{rep}_in_set'].sum()),
                   cert_correct_hybrid_wrong=int((cc[f'hyb{rep}_wrong'] == 1).sum()), cert_correct_n=len(cc)))
pd.DataFrame(hy).to_csv(TAB / 'C_hybrid_exploratory.csv', index=False)
drivers = cw.error_driver.value_counts().to_dict()

meta = dict(
            inputs={str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                    [EXP1 / 'results/per_patient_states.csv', P1 / 'results/evidence.json', P1 / 'results/reference_HIDDEN.json', HYB]},
            primary=PRIMARY, bootstrap=dict(n=B_N, seed=SEED), A=pt, A_p=ta_p, B=bp,
            C=dict(n_cert_wrong=len(cw), drivers=drivers, residual_auroc=auc, residual_auroc_ci=pct(np.array(ab)),
                   residual_missing_cert_wrong=int(cw.residual_mm.isna().sum())))
(RES / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
print(json.dumps({k: v for k, v in meta.items() if k != 'inputs'}, indent=1, default=float))
print(ta.round(4).to_string()); print(tb); print(pd.DataFrame(rows_b).round(4).to_string())
print(pd.DataFrame(desc)[['group', 'n', 'ref_included', 'any_ref_pair_beyond_gate', 'any_spurious_new', 'residual_mm_median', 'ref_excess_over_q_median', 'max_ref_pair_mm_median']].to_string())
print(cw[keep[:12]].sort_values('residual_mm').to_string()); print(pd.DataFrame(hy).to_string())
