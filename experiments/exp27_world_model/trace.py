"""Experiment 2.7: per-prediction transition traces, support/OOD checks and error decomposition.

For each patient: repeat-averaged OOF predictions (gradient boosting and Elastic Net, WM-0 and WM-3), conformal
interval and P(ORR) from uncertainty.py, input state + provenance (patient_states/), action, horizon, missing evidence,
support checks (computed against the outer TRAINING patients of each fold, flagged if >= 3 of 5 repeats), and for
Elastic Net WM-3 the exact per-feature contributions (coefficient x standardized value) to the predicted change.
Major errors (|error| >= 50 points or rule-layer ORR misclassified, gradient boosting WM-3) are assigned tags by fixed
precedence and get a case report. Usage: trace.py <model run dir>
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors

import audit_transitions as A
import fit_models as FM

PRECEDENCE = ['STATE_INPUT_ERROR', 'MISSING_EVIDENCE', 'OUT_OF_DISTRIBUTION', 'TIME_HORIZON_ERROR', 'ACTION_ENCODING_ERROR',
              'RESPONSE_RULE_ERROR', 'OTHER', 'FUTURE_MEASUREMENT_ERROR']
MODEL_VERSION = 'exp27 GB/EN, locked splits'


def fold_support(fr, fold):
    """OOD / support checks for test patients, relative to the outer training patients only."""
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    cols = FM.VARIANTS['WM-3']
    tr, te = fold['train'], fold['test']
    est, grid = FM.regressor('enet')
    pos = {p: i for i, p in enumerate(tr)}
    inner = [(np.array([pos[p] for p in f['train']]), np.array([pos[p] for p in f['valid']])) for f in fold['inner']]
    gs = GridSearchCV(est, grid, cv=inner, scoring='neg_mean_absolute_error', n_jobs=1).fit(fr.loc[tr, cols], fr.loc[tr, 'y_pct'])
    pipe = gs.best_estimator_
    imp, sc, en = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    Ztr = sc.transform(imp.transform(fr.loc[tr, cols]))
    Zte = sc.transform(imp.transform(fr.loc[te, cols]))
    names = list(cols) + [f'missing:{cols[i]}' for i in imp.indicator_.features_] if imp.indicator_ is not None else list(cols)
    nn = NearestNeighbors(n_neighbors=6).fit(Ztr)
    d_tr = nn.kneighbors(Ztr)[0][:, 1:].mean(1)            # leave-one-out 5-NN distance among training patients
    thr = np.percentile(d_tr, 95)
    d_te = nn.kneighbors(Zte, n_neighbors=5)[0].mean(1)
    lo, hi = fr.loc[tr, cols].min(), fr.loc[tr, cols].max()
    dt_lo, dt_hi = np.percentile(fr.loc[tr, 'dt_weeks'], [5, 95])
    arm_n = fr.loc[tr, 'deb_tace'].value_counts().to_dict()
    out = []
    for i, p in enumerate(te):
        x = fr.loc[p, cols]
        outside = [c for c in cols if pd.notna(x[c]) and (x[c] < lo[c] or x[c] > hi[c])]
        contrib = dict(zip(names, (en.coef_ * Zte[i]).tolist()))
        out.append(dict(patient_id=p, repeat=fold['repeat'], knn_dist=float(d_te[i]), knn_thr=float(thr), knn_ood=bool(d_te[i] > thr),
                        range_outside=';'.join(outside), dt_outside_p5_p95=bool(x['dt_weeks'] < dt_lo or x['dt_weeks'] > dt_hi),
                        arm_train_n=int(arm_n.get(int(x['deb_tace']), 0)), en_intercept=float(en.intercept_), en_contrib=json.dumps(contrib)))
    return out


def main():
    run = Path(sys.argv[1])
    sp = json.loads((FM.SPLITS / 'splits.json').read_text())
    _, fr = FM.state.build()
    sup = pd.DataFrame([r for rs in Parallel(n_jobs=64)(delayed(fold_support)(fr, f) for f in sp['folds']) for r in rs])
    unc = pd.read_csv(run / 'uncertainty/intervals_per_repeat.csv.gz')
    ua = unc.groupby(['model', 'variant', 'patient_id']).agg(pred=('y_pred', 'mean'), lo50=('lo50', 'mean'), hi50=('hi50', 'mean'),
                                                            lo90=('lo90', 'mean'), hi90=('hi90', 'mean'), p_orr=('p_orr', 'mean')).reset_index()
    man = pd.read_csv(FM.state.AUDIT_RUN / 'transition_manifest.csv').set_index('patient_id')
    prop = pd.read_csv(FM.state.AUDIT_RUN / 'propensity_scores.csv').set_index('patient_id')
    ps_lo, ps_hi = json.loads((FM.state.AUDIT_RUN / 'audit_summary.json').read_text())['propensity']['common_support']
    states = {p.stem: json.loads(p.read_text()) for p in (run / 'patient_states').glob('*.json')}
    out = run / 'traces'
    (out / 'patients').mkdir(parents=True, exist_ok=True)
    (out / 'case_reports').mkdir(exist_ok=True)
    rows = []
    for p in sorted(fr.index):
        s, r, m = sup[sup.patient_id == p], fr.loc[p], man.loc[p]
        contrib = pd.DataFrame([json.loads(c) for c in s.en_contrib]).mean()
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
        pr = {f'{mo}_{v}': ua[(ua.model == mo) & (ua.variant == v) & (ua.patient_id == p)].iloc[0] for mo in ('gbr', 'enet') for v in ('WM-0', 'WM-3')}
        g = pr['gbr_WM-3']
        pred_fu = max(r.bl_mrecist_mm * (1 + g.pred / 100), 0)
        readers = [r[f'y_pct_r{k}'] for k in (1, 2, 3)]
        rv = [x for x in readers if pd.notna(x)]
        support = dict(knn_ood_repeats=int(s.knn_ood.sum()), range_outside=sorted(set(';'.join(s.range_outside).split(';')) - {''}),
                       dt_outside_train_p5_p95_repeats=int(s.dt_outside_p5_p95.sum()), action_arm_train_n_min=int(s.arm_train_n.min()),
                       propensity_DEB=float(prop.at[p, 'propensity_DEB_cv']), propensity_in_common_support=bool(ps_lo <= prop.at[p, 'propensity_DEB_cv'] <= ps_hi))
        ood = support['knn_ood_repeats'] >= 3 or bool(support['range_outside']) or not support['propensity_in_common_support']
        missing = [k for k, v in states[p]['missingness'].items() if v]
        flags = str(m.data_quality_flags) if pd.notna(m.data_quality_flags) else ''
        err = g.pred - r.y_pct
        mis = int(g.pred <= -30) != int(r.orr)
        tags = []
        if any(f in flags for f in ('seg_reference_resolved_by_geometry', 'frame_of_reference_uid_mismatch')):
            tags.append('STATE_INPUT_ERROR')
        if missing:
            tags.append('MISSING_EVIDENCE')
        if ood:
            tags.append('OUT_OF_DISTRIBUTION')
        if support['dt_outside_train_p5_p95_repeats'] >= 3:
            tags.append('TIME_HORIZON_ERROR')
        if m.treatment_raw in ('Cisplatin, Mitomycin-C', 'Cisplastin'):
            tags.append('ACTION_ENCODING_ERROR')  # non-standard regimen folded into cTACE
        # the rule layer is verified exact on true measurements (96/96, 95/95), so RESPONSE_RULE_ERROR cannot occur by construction
        if len(rv) >= 2 and (max(rv) - min(rv) >= 30 or len({x <= -30 for x in rv}) > 1):
            tags.append('OTHER')  # ground-truth reader disagreement (label uncertainty)
        elif len(rv) < 2:
            tags.append('OTHER')  # single-reader label
        tags.append('FUTURE_MEASUREMENT_ERROR')
        major = abs(err) >= 50 or mis
        trace = {
            'patient_id': p, 'input_state_version': FM.state.STATE_SCHEMA_VERSION, 'model_version': MODEL_VERSION,
            'transition': 'S_BL + A_TACE + dt_BL->FU -> S_FU',
            'baseline_evidence': {sec: states[p][sec] for sec in ('disease', 'imaging', 'clinical')},
            'provenance': states[p]['provenance'], 'missing_evidence': missing,
            'action': {'treatment_normalized': m.treatment_normalized, 'treatment_raw': m.treatment_raw},
            'time_horizon': {'dt_weeks': r.dt_weeks, 'definition': 'baseline CT -> follow-up CT (DICOM StudyDate); not time since TACE'},
            'prediction': {'model': 'gradient boosting WM-3 (S + A + dt)', 'pred_mrecist_change_pct': round(g.pred, 1),
                           'pi50': [round(g.lo50, 1), round(g.hi50, 1)], 'pi90': [round(g.lo90, 1), round(g.hi90, 1)],
                           'pred_followup_mrecist_mm': round(pred_fu, 1), 'p_orr': round(g.p_orr, 3),
                           'rule_layer_response': 'ORR (CR/PR)' if g.pred <= -30 else 'non-ORR (SD/PD)',
                           'uncertain': bool(0.2 < g.p_orr < 0.8), 'pi90_straddles_threshold': bool(g.lo90 <= -30 <= g.hi90),
                           'other_models': {k: round(v.pred, 1) for k, v in pr.items()}},
            'explanation_enet_WM3': {'intercept': round(float(s.en_intercept.mean()), 2),
                                     'top_contributions_pct_points': {k: round(v, 2) for k, v in top.items()}},
            'support': support, 'out_of_distribution': ood,
            'observed': {'mrecist_change_pct': round(r.y_pct, 1), 'followup_mrecist_mm': r.y_fu_mm, 'consensus_response': r.consensus_category,
                         'reader_change_pct': [None if pd.isna(x) else round(x, 1) for x in readers]},
            'error': {'pct_points': round(err, 1), 'orr_misclassified': mis, 'major': major, 'tags': tags, 'primary': tags[0]},
            'data_quality_flags': flags}
        (out / 'patients' / f'{p}.json').write_text(json.dumps(trace, indent=1, default=float))
        rows.append(dict(patient_id=p, arm=m.treatment_normalized, dt_weeks=r.dt_weeks, y=r.y_pct, pred=g.pred, err=err, abs_err=abs(err),
                         p_orr=g.p_orr, orr=int(r.orr), orr_misclassified=mis, major=major, primary=tags[0], tags=';'.join(tags), ood=ood,
                         uncertain=0.2 < g.p_orr < 0.8, knn_ood_repeats=support['knn_ood_repeats']))
        if major:
            write_case(out / 'case_reports' / f'{p}.md', trace)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'trace_summary.csv', index=False)
    maj = df[df.major]
    dec = pd.DataFrame([dict(category=c, primary_n=int((maj.primary == c).sum()), any_n=int(maj.tags.str.contains(c).sum()),
                             all_patients_any_n=int(df.tags.str.contains(c).sum())) for c in PRECEDENCE])
    dec.to_csv(out / 'error_decomposition.csv', index=False)
    print(f'major errors: {len(maj)} / {len(df)} (|err|>=50: {int((df.abs_err >= 50).sum())}, ORR misclassified: {int(df.orr_misclassified.sum())})')
    print(dec.to_string(index=False))
    print('OOD patients:', df.ood.sum(), '| MAE OOD vs in-support: %.1f vs %.1f' % (df[df.ood].abs_err.mean(), df[~df.ood].abs_err.mean()))
    print('uncertain (0.2<P<0.8):', int(df.uncertain.sum()), '| misclassified among uncertain / confident:',
          int(df[df.uncertain].orr_misclassified.sum()), '/', int(df[~df.uncertain].orr_misclassified.sum()))
    print(maj.sort_values('abs_err', ascending=False).head(12)[['patient_id', 'arm', 'y', 'pred', 'p_orr', 'orr', 'tags']].round(1).to_string(index=False))


def write_case(path, t):
    pr, ob, e = t['prediction'], t['observed'], t['error']
    ev = t['baseline_evidence']
    lines = [f"# Case {t['patient_id']} — {e['primary']}", '',
             f"Tags: {', '.join(e['tags'])}. Error {e['pct_points']:+.1f} points; ORR misclassified: {e['orr_misclassified']}.", '',
             '| | Value |', '|---|---|',
             f"| Treatment | {t['action']['treatment_normalized']} ({t['action']['treatment_raw']}) |",
             f"| Horizon | {t['time_horizon']['dt_weeks']} weeks (BL CT → FU CT) |",
             f"| Baseline mRECIST / RECIST | {ev['disease']['bl_mrecist_mm']} / {ev['disease']['bl_recist_mm']} mm |",
             f"| SEG mass volume | {10 ** ev['imaging']['log_seg_mass_ml']:.1f} mL (tumour/liver {ev['imaging']['seg_mass_to_liver']:.2f}) |",
             f"| Child-Pugh B/C, BCLC | {ev['clinical']['child_pugh_b_or_c']}, {ev['clinical']['bclc_stage']} |",
             f"| Predicted change (GB WM-3) | {pr['pred_mrecist_change_pct']} % (50 % PI {pr['pi50']}, 90 % PI {pr['pi90']}) |",
             f"| P(ORR) / rule layer | {pr['p_orr']} / {pr['rule_layer_response']}; uncertain: {pr['uncertain']} |",
             f"| Other models | {pr['other_models']} |",
             f"| Observed change | {ob['mrecist_change_pct']} % ({ob['consensus_response']}); readers {ob['reader_change_pct']} |",
             f"| Support | kNN-OOD in {t['support']['knn_ood_repeats']}/5 repeats; outside training range: {t['support']['range_outside'] or 'none'}; "
             f"propensity {t['support']['propensity_DEB']:.2f} (common support: {t['support']['propensity_in_common_support']}) |",
             f"| Missing evidence | {t['missing_evidence'] or 'none'} |",
             f"| Data-quality flags | {t['data_quality_flags'] or 'none'} |", '',
             'Largest Elastic Net WM-3 contributions (points of predicted change): ' +
             ', '.join(f'{k} {v:+.1f}' for k, v in t['explanation_enet_WM3']['top_contributions_pct_points'].items())]
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
