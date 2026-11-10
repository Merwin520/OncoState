"""Lock repeated nested 5-fold patient-level CV splits and audit them.

Outer: 5 repeats x StratifiedKFold(5) on strata = treatment (cTACE / DEB-TACE) x consensus response bin (CR / PR / non-ORR);
the response bins are coarse bins of mRECIST % change (-100 %, (-100, -30] %, > -30 %).
Per repeat, the first seed (in a fixed order) whose 5 test folds all pass the balance criteria is used (first-feasible, not best).
Inner: StratifiedKFold(3) on treatment x ORR, drawn from the outer training patients only.
Writes outputs/exp27/splits/splits.json (+ sha256) and reports/split_audit.md.
The audit's pass criteria are fixed here, before any model is fitted; fit scripts refuse to run on a failed audit.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import state

REPO = state.REPO
MAX_SEEDS_PER_REPEAT = 10000
N_REPEATS, N_OUTER, N_INNER = 5, 5, 3
CRITERIA = dict(min_deb_per_test_fold=4, min_non_orr_per_test_fold=4, min_cr_per_test_fold=1, max_abs_smd_test_vs_train=0.5)


def make_repeat(fr, rep, seed):
    strata = fr.deb_tace.astype(str) + '_' + np.where(fr.consensus_category == 'CR', 'CR', np.where(fr.orr == 1, 'PR', 'nonORR'))
    ids = fr.index.to_numpy()
    folds = []
    for k, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=seed).split(ids, strata)):
        tr_ids = ids[tr]
        inner_strata = (fr.loc[tr_ids, 'deb_tace'].astype(str) + '_' + fr.loc[tr_ids, 'orr'].astype(str)).to_numpy()
        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=1000 + 10 * rep + k)
        folds.append(dict(repeat=rep, fold=k, outer_seed=seed, train=sorted(tr_ids.tolist()), test=sorted(ids[te].tolist()),
                          inner=[dict(train=sorted(tr_ids[a].tolist()), valid=sorted(tr_ids[b].tolist()))
                                 for a, b in inner.split(tr_ids, inner_strata)]))
    return folds


def make(fr):
    """Per repeat, the first seed in rep*MAX_SEEDS_PER_REPEAT + 0, 1, 2, ... whose folds pass every audit criterion."""
    folds, tried, seen = [], {}, set()
    for rep in range(N_REPEATS):
        for j in range(MAX_SEEDS_PER_REPEAT):
            cand = make_repeat(fr, rep, rep * MAX_SEEDS_PER_REPEAT + j)
            key = tuple(tuple(f['test']) for f in sorted(cand, key=lambda f: f['test']))
            if key not in seen and not audit(fr, cand, repeats=[rep])[1]:
                seen.add(key)
                folds += cand
                tried[rep] = j + 1
                break
        else:
            sys.exit(f'repeat {rep}: no seed passes the audit criteria')
    return folds, tried


def smd(a, b):
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def q(x):
    return f'{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]'


def audit(fr, folds, repeats=range(N_REPEATS)):
    rows, fails = [], []
    for rep in repeats:
        tests = [f['test'] for f in folds if f['repeat'] == rep]
        allt = sum(tests, [])
        if sorted(allt) != sorted(fr.index) or len(allt) != len(set(allt)):
            fails.append(f'repeat {rep}: test folds do not partition the cohort')
    for f in folds:
        te, tr = fr.loc[f['test']], fr.loc[f['train']]
        if set(f['test']) & set(f['train']):
            fails.append(f"r{f['repeat']}f{f['fold']}: train/test overlap")
        for inn in f['inner']:
            if not set(inn['train'] + inn['valid']) <= set(f['train']) or set(inn['train']) & set(inn['valid']):
                fails.append(f"r{f['repeat']}f{f['fold']}: inner fold leaves the outer training set")
        sm = {k: smd(te[k], tr[k]) for k in ('y_pct', 'bl_mrecist_mm', 'log_seg_mass_ml', 'dt_weeks', 'child_pugh_b_or_c', 'deb_tace')}
        r = dict(repeat=f['repeat'], fold=f['fold'], N=len(te), cTACE=int((te.deb_tace == 0).sum()), DEB=int(te.deb_tace.sum()),
                 ORR=int(te.orr.sum()), nonORR=int((te.orr == 0).sum()), CR=int((te.consensus_category == 'CR').sum()),
                 PD=int((te.consensus_category == 'PD').sum()), pct_change=q(te.y_pct), bl_mrecist_mm=q(te.bl_mrecist_mm),
                 seg_mass_ml=q(10 ** te.log_seg_mass_ml), child_pugh_A=int((te.child_pugh_b_or_c == 0).sum()),
                 child_pugh_BC=int(te.child_pugh_b_or_c.sum()), dt_weeks=q(te.dt_weeks),
                 max_abs_smd=max(abs(v) for v in sm.values()), max_smd_var=max(sm, key=lambda k: abs(sm[k])))
        rows.append(r)
        tag = f"r{f['repeat']}f{f['fold']}"
        if r['DEB'] < CRITERIA['min_deb_per_test_fold'] or r['cTACE'] < CRITERIA['min_deb_per_test_fold']:
            fails.append(f'{tag}: too few patients in one treatment arm')
        if r['nonORR'] < CRITERIA['min_non_orr_per_test_fold'] or r['ORR'] < CRITERIA['min_non_orr_per_test_fold']:
            fails.append(f'{tag}: too few patients in one ORR class')
        if r['CR'] < CRITERIA['min_cr_per_test_fold']:
            fails.append(f'{tag}: no CR')
        if r['max_abs_smd'] > CRITERIA['max_abs_smd_test_vs_train']:
            fails.append(f"{tag}: |SMD| {r['max_abs_smd']:.2f} on {r['max_smd_var']}")
    return pd.DataFrame(rows), fails


def main():
    _, fr = state.build()
    out = REPO / 'outputs/exp27/splits'
    if (out / 'splits.json').exists():
        sys.exit(f'{out} exists; splits are locked.')
    folds, tried = make(fr)
    tab, fails = audit(fr, folds)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(cohort=sorted(fr.index.tolist()), n_repeats=N_REPEATS, n_outer=N_OUTER, n_inner=N_INNER,
                   stratification='treatment x {CR, PR, non-ORR} (outer); treatment x ORR (inner)', criteria=CRITERIA,
                   audit_passed=not fails, audit_failures=fails, seeds_tried_per_repeat=tried, folds=folds)
    blob = json.dumps(payload, indent=1)
    (out / 'splits.json').write_text(blob)
    (out / 'splits.sha256').write_text(hashlib.sha256(blob.encode()).hexdigest() + '\n')
    tab.to_csv(out / 'split_audit.csv', index=False)
    write_report(fr, tab, fails, out, tried)
    print('audit', 'PASSED' if not fails else f'FAILED: {fails}')


def write_report(fr, tab, fails, out, tried):
    cols = ['repeat', 'fold', 'N', 'cTACE', 'DEB', 'ORR', 'nonORR', 'CR', 'PD', 'pct_change', 'bl_mrecist_mm', 'seg_mass_ml',
            'child_pugh_A', 'child_pugh_BC', 'dt_weeks', 'max_abs_smd', 'max_smd_var']
    t = tab[cols].copy()
    t['max_abs_smd'] = t.max_abs_smd.round(2)
    lines = ['# Split audit', '',
             f"Splits: `{out.relative_to(REPO)}/splits.json` (sha256 in `splits.sha256`), locked before any model fit. "
             f"Script: `experiments/exp27_world_model/splits.py`.", '',
             f"**Audit: {'PASSED' if not fails else 'FAILED'}**" + ('' if not fails else ' — ' + '; '.join(fails)), '',
             '## Design', '',
             f'- Cohort: {len(fr)} primary transitions (cTACE {int((fr.deb_tace == 0).sum())}, DEB-TACE {int(fr.deb_tace.sum())}; '
             f'ORR {int(fr.orr.sum())} / non-ORR {int((fr.orr == 0).sum())} — verified: ORR = mRECIST change ≤ −30 % equals rule-layer CR+PR for all 82).',
             f'- Outer: {N_REPEATS} repeats × stratified {N_OUTER}-fold, strata = treatment × {{CR, PR, non-ORR}} '
             '(the response bins are coarse bins of mRECIST % change: −100 %, (−100, −30] %, > −30 %). Smallest stratum: DEB-TACE CR, n = 5.',
             f'- Inner: stratified {N_INNER}-fold on treatment × ORR inside each outer training set (for hyperparameters only).',
             f'- Balance: per repeat, the first stratified seed (fixed order) whose 5 test folds all meet the criteria below was used '
             f'(candidates tried per repeat: {tried}). First-feasible, not best-of-N. Outcome-blind with respect to any model.',
             '- Every model and variant (naive baselines, WM-0…WM-3, WM-shuffle, reader sensitivity) uses these identical folds.',
             '- Imputation, scaling and tuning are fitted inside training folds only (enforced in the fit script by sklearn pipelines).', '',
             '## Pass criteria (fixed before fitting)', '',
             '- each repeat: test folds partition the 82 patients; no train/test overlap; inner folds stay inside the outer training set',
             f"- every test fold: ≥ {CRITERIA['min_deb_per_test_fold']} patients per treatment arm, ≥ {CRITERIA['min_non_orr_per_test_fold']} per ORR class, ≥ {CRITERIA['min_cr_per_test_fold']} CR",
             f"- |standardized difference| test vs train ≤ {CRITERIA['max_abs_smd_test_vs_train']} for mRECIST % change, baseline mRECIST, SEG mass volume, Δt, Child-Pugh B/C, treatment", '',
             '## Per outer fold', '',
             'Continuous columns: median [IQR]. pct_change = mRECIST % change (target). dt_weeks = BL CT → FU CT (not treatment → FU).', '',
             t.to_markdown(index=False), '',
             '## Range across the 25 test folds', '',
             f"- N {tab.N.min()}–{tab.N.max()}; DEB-TACE {tab.DEB.min()}–{tab.DEB.max()}; non-ORR {tab.nonORR.min()}–{tab.nonORR.max()}; "
             f"CR {tab.CR.min()}–{tab.CR.max()}; Child-Pugh B/C {tab.child_pugh_BC.min()}–{tab.child_pugh_BC.max()}; "
             f"PD {tab.PD.min()}–{tab.PD.max()} (PD n = 3 overall, descriptive only).",
             f"- Largest |SMD| test vs train: {tab.max_abs_smd.max():.2f} ({tab.loc[tab.max_abs_smd.idxmax(), 'max_smd_var']}).", '']
    (REPO / 'reports').mkdir(exist_ok=True)
    (REPO / 'reports/split_audit.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
