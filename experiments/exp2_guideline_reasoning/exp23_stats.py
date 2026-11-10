"""Experiment 2.3 statistics: real HCUCH (patient-level bootstrap, 5000), synthetic benchmark (Wilson / case bootstrap, benchmark uncertainty
only), LLM repeats (mean +- SD, per-case consistency), paired differences (paired bootstrap, exact McNemar, Benjamini-Hochberg)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

import exp23_rules as RL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
NB, SEED = 5000, 20260921
LABELS = RL.RESPONSES
SYSTEMS = ['Full OncoAgent', 'No-retrieval', 'LLM-only', 'Rule executor only']
rng = np.random.default_rng(SEED)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def cls_metrics(y, p):
    """5 core multiclass metrics over the RECIST response labels (CR/PR/SD/PD/NE): macro precision, macro recall, macro F1, weighted F1, Cohen's kappa.
    Macro precision/recall/F1 are restricted to labels present in the reference (y) so that response categories absent from a given
    case set do not silently count as 0 and drag the macro average down; weighted F1 uses the full label set and reference support.

    A system can emit something outside the 5-label vocabulary (e.g. 'ERROR' from Rule-executor-only crashing on missing evidence,
    'PARSE_ERROR' from an LLM/VLM whose JSON failed to parse). precision/recall/f1_score handle this correctly on their own (an
    out-of-vocabulary prediction is simply never equal to any real label, so it is counted as a false negative for whatever the true
    label was). cohen_kappa_score does NOT: it builds sklearn's confusion_matrix restricted to `labels`, which silently DROPS every
    row whose prediction is not in that list from the matrix entirely, rather than counting it as a disagreement -- shrinking N and
    inflating kappa. Any such value is added to the label set passed to cohen_kappa_score so it is scored as (yet another way of
    being) wrong, not excluded."""
    y, p = list(y), list(p); present = [l for l in LABELS if l in set(y)]; oov = sorted(set(p) | set(y) - set(LABELS))
    return dict(macro_precision=precision_score(y, p, labels=present, average='macro', zero_division=0), macro_recall=recall_score(y, p, labels=present, average='macro', zero_division=0),
                macro_f1=f1_score(y, p, labels=present, average='macro', zero_division=0), weighted_f1=f1_score(y, p, labels=LABELS, average='weighted', zero_division=0),
                kappa=cohen_kappa_score(y, p, labels=LABELS + oov) if len(set(y)) > 1 or len(set(p)) > 1 else np.nan)


def case_matrix(df, col):
    """cases x repeats matrix of a boolean column (deterministic systems: one column)."""
    return df.pivot_table(index='case_id', columns='repeat_id', values=col, aggfunc='first')


def boot_mean(v, nb=NB):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), size=(nb, len(v))); m = v[idx].mean(1); return np.percentile(m, [2.5, 97.5])


def paired(a, b, nb=NB):
    """paired bootstrap of mean(a-b) over cases; exact McNemar on the binarised outcomes (majority of repeats >=0.5)."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    idx = rng.integers(0, len(d), size=(nb, len(d))); bm = d[idx].mean(1)
    A, B = a >= .5, b >= .5; n10, n01 = int((A & ~B).sum()), int((~A & B).sum())
    p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
    return d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5), p, n10, n01


def bh(ps):
    ps = np.asarray(ps, float); n = len(ps); o = np.argsort(ps); q = np.empty(n); prev = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        prev = min(prev, ps[i] * n / rank); q[i] = prev
    return q


def main():
    real = pd.read_csv(OUT / 'real_hcuch_cases.csv'); pred = pd.read_csv(OUT / 'benchmark_predictions_long.csv'); cases = pd.read_csv(OUT / 'rule_challenge_cases.csv')
    stats = []
    def S(section, metric, system, est, lo=np.nan, hi=np.nan, n=np.nan, note=''):
        stats.append(dict(section=section, metric=metric, system=system, estimate=est, ci95_lo=lo, ci95_hi=hi, n=n, note=note))

    # ---------------- A. real HCUCH -----------------
    pats = real.patient_id.unique(); g = {p: real[real.patient_id == p] for p in pats}
    def rmet(df):
        return dict(accuracy=df.correct.mean(), rule_correct=df.rule_correct.mean(), trace=df.traceability_complete.mean(), overall_correct=df.overall_correct.mean(),
                    mask_agree=df.mask_response_agrees_with_expert.mean())
    pt = rmet(real); bs = {k: [] for k in pt}
    for _ in range(NB):
        d = pd.concat([g[p] for p in rng.choice(pats, len(pats))]); m = rmet(d)
        for k in bs: bs[k].append(m[k])
    for k, v in pt.items():
        S('real_hcuch', k, 'Full OncoAgent', v, *np.percentile(bs[k], [2.5, 97.5]), n=len(real), note=f'patient-level bootstrap {NB}; {len(pats)} patients')
    cm = cls_metrics(real.reference_response, real.predicted_response); [S('real_hcuch', k, 'Full OncoAgent', v, n=len(real), note='point estimate') for k, v in cm.items()]
    mk = cls_metrics(real.reference_response, real.mask_derived_response)
    for k, v in mk.items(): S('real_hcuch', 'mask_derived_' + k, 'state-side measurement', v, n=len(real), note='response from mask-derived diameters vs response from expert measurements')
    for cl, n in real.reference_response.value_counts().items(): S('real_hcuch', f'reference_count_{cl}', 'reference', n, n=len(real))

    # ---------------- B. synthetic main ---------------
    main_ = pred[pred.subset == 'main']; rows = []
    for sysn in SYSTEMS:
        d = main_[main_.system == sysn]; reps = sorted(d.repeat_id.unique()); ncase = d.case_id.nunique()
        per_rep = []
        for r in reps:
            x = d[d.repeat_id == r]; m = cls_metrics(x.reference_response, x.predicted_response)
            ab = x[x.expected_abstention]; nab = x[~x.expected_abstention]; cit = x.dropna(subset=['citation_present'])
            per_rep.append(dict(accuracy=x.correct.mean(), acc_complete=nab.correct.mean(), rule=x.rule_clause_correct.mean(), trace=x.traceability_complete.mean(), **m,
                                abstention_correct=(ab.predicted_response == 'NE').mean(), unsupported=ab.unsupported_decision.mean(), false_abstain=(nab.predicted_response == 'NE').mean(),
                                cit_present=cit.citation_present.astype(float).mean() if len(cit) else np.nan, cit_correct=cit.citation_correct.astype(float).mean() if len(cit) and cit.citation_correct.notna().any() else np.nan,
                                cit_supported=cit.citation_supported.astype(float).mean() if len(cit) else np.nan))
        pr = pd.DataFrame(per_rep); mean, sd = pr.mean(), pr.std(ddof=0)
        colmap = {'accuracy': 'correct', 'acc_complete': 'correct', 'rule': 'rule_clause_correct', 'trace': 'traceability_complete', 'cit_supported': 'citation_supported'}
        cis = {}
        for met, col in colmap.items():
            dd_ = d[d[col].notna()]
            if met == 'acc_complete': dd_ = dd_[~dd_.expected_abstention]
            cm_ = case_matrix(dd_, col).astype(float).mean(1).values
            if len(cm_) == 0: cis[met] = (np.nan, np.nan); continue
            cis[met] = wilson(int(round(cm_.sum())), len(cm_)) if len(reps) == 1 else tuple(boot_mean(cm_))
        for met in pr.columns:
            S('synthetic_main', met, sysn, mean[met], *cis.get(met, (np.nan, np.nan)), n=ncase, note=(f'mean over {len(reps)} repeats; SD {sd[met]:.4f}; case bootstrap CI' if len(reps) > 1 else 'exact; Wilson CI is benchmark uncertainty'))
        if len(reps) > 1:
            rm = case_matrix(d, 'predicted_response'); cons = (rm.nunique(axis=1) == 1).mean(); maj = (case_matrix(d, 'correct').astype(float).mean(1) > .5).mean()
            S('synthetic_main', 'per_case_consistency', sysn, cons, n=ncase, note='all repeats identical'); S('synthetic_main', 'majority_vote_accuracy', sysn, maj, n=ncase)
        for ct, x in d.groupby('challenge_type'):
            S('synthetic_by_type', f'accuracy|{ct}', sysn, x.correct.mean(), n=x.case_id.nunique(), note='repeat-level mean')

    # ---------------- C. paired differences -----------------
    def per_case(sysn, col, mask=None):
        d = main_[main_.system == sysn]
        if mask is not None: d = d[d.case_id.isin(mask)]
        return case_matrix(d, 'correct' if col == 'correct_complete_evidence' else col).astype(float).mean(1)
    cit_cases = set(main_[(main_.system == 'Full OncoAgent') & main_.citation_present.notna()].case_id)
    complete_cases = set(main_[(main_.system == 'Full OncoAgent') & (~main_.expected_abstention)].case_id)
    comps = []
    for other in ['No-retrieval', 'LLM-only', 'Rule executor only']:
        for met in ['correct', 'correct_complete_evidence', 'rule_clause_correct', 'traceability_complete', 'citation_supported']:
            if met == 'citation_supported' and other == 'Rule executor only': continue
            mask = cit_cases if met == 'citation_supported' else (complete_cases if met == 'correct_complete_evidence' else None)
            a, b = per_case('Full OncoAgent', met, mask), per_case(other, met, mask); idx = a.index.intersection(b.index)
            if len(idx) == 0 or a.loc[idx].isna().all(): continue
            dm, lo, hi, p, n10, n01 = paired(a.loc[idx].fillna(0).values, b.loc[idx].fillna(0).values)
            comps.append((f'Full - {other}', met, dm, lo, hi, p, n10, n01, len(idx)))
    q = bh([c[5] for c in comps])
    for c, qq in zip(comps, q):
        S('paired_difference', c[1], c[0], c[2], c[3], c[4], n=c[8], note=f'McNemar exact p={c[5]:.3g}; BH q={qq:.3g}; Full-only-right={c[6]}, other-only-right={c[7]}')

    # ---------------- D. threshold stress -----------------
    st = pred[pred.subset == 'threshold_stress'].merge(cases[['case_id', 'series_id', 'series_pos', 'stress_family', 'distance_to_threshold', 'change_baseline_pct', 'change_nadir_pct', 'absolute_change_mm', 'new_lesion']], on='case_id')
    st = st.rename(columns={'reference_response': 'expected_response', 'predicted_response': 'predicted'})
    st[['case_id', 'system', 'repeat_id', 'series_id', 'series_pos', 'stress_family', 'distance_to_threshold', 'change_baseline_pct', 'change_nadir_pct', 'absolute_change_mm', 'new_lesion',
        'expected_response', 'predicted', 'correct']].to_csv(OUT / 'threshold_stress_results.csv', index=False)
    RANK = {'PR': 0, 'SD': 1, 'PD': 2}
    for sysn in SYSTEMS:
        d = st[st.system == sysn]
        if d.empty: continue
        for fam, x in d.groupby('stress_family'):
            reps = sorted(x.repeat_id.unique()); sw, pc, mv, acc = [], [], [], []
            for r in reps:
                y = x[x.repeat_id == r]; ok = []; pcs = []; mvs = []
                for sid, s in y.groupby('series_id'):
                    s = s.sort_values('series_pos'); ok.append(bool(s.correct.all()))
                    pcs += [bool(s.correct.iloc[i] and s.correct.iloc[i + 1]) for i in range(len(s) - 1)]
                    if fam in ('S1_PR', 'S2_PD', 'S3_ABS'):
                        rk = [RANK.get(v, 9) for v in s.predicted]; mvs.append(any(rk[j] < rk[i] for i in range(len(rk)) for j in range(i + 1, len(rk))))
                sw.append(np.mean(ok)); pc.append(np.mean(pcs)); mv.append(np.mean(mvs) if mvs else np.nan); acc.append(y.correct.mean())
            n_series = x.series_id.nunique()
            name = {'S4_NEW': 'new_lesion_override_correctness', 'S5_NADIR': 'nadir_use_correctness', 'S3_ABS': 'absolute_increase_rule_correctness'}.get(fam, 'threshold_switch_accuracy')
            S('threshold_stress', f'{name}|{fam}', sysn, np.mean(sw), n=n_series, note='all points of a matched series/pair correct')
            S('threshold_stress', f'pairwise_consistency|{fam}', sysn, np.mean(pc), n=n_series); S('threshold_stress', f'point_accuracy|{fam}', sysn, np.mean(acc), n=len(x) // len(reps))
            if not np.isnan(np.nanmean(mv) if len(mv) else np.nan): S('threshold_stress', f'monotonicity_violation_rate|{fam}', sysn, np.nanmean(mv), n=n_series)
    pd.DataFrame(stats).to_csv(OUT / 'exp23_statistics.csv', index=False)
    print(len(stats), 'statistics rows'); return pd.DataFrame(stats)


if __name__ == '__main__':
    s = main(); pd.set_option('display.width', 200)
    print(s[s.section == 'synthetic_main'][['metric', 'system', 'estimate', 'ci95_lo', 'ci95_hi']].round(3).to_string(index=False))
