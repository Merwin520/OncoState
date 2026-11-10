#!/usr/bin/env python3
"""Experiment 4 source data (no plotting): builds figures/figure5_data.csv from results/ and the longitudinal-forecasting analysis.

    python figure5_data.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from common import FIG, REPO, RES, T2

R29 = REPO / 'outputs/longitudinal_forecasting/analysis'
SYS = ['oncostate', 'full_history', 'memory', 'uncorrected']
TR = ['T0->T1', 'T1->T2', 'T2->T3']


def build():
    R = []
    def r(panel, series, label, value, lo=np.nan, hi=np.nan, n=np.nan, x=np.nan, src=''):
        R.append(dict(panel=panel, series=series, label=label, value=value, lo=lo, hi=hi, n=n, x=x, src=src))
    M = pd.read_csv(RES / '4A_metrics.csv'); M = M[M.subgroup == 'all']
    for t in TR:
        for task, s in ((f'{t}|last|locf', 'locf'), (f'{t}|state|mean', 'mean'), (f'{t}|state|ridge', 'ridge'), (f'{t}|state|gbm', 'gbm'), (f'{t}|last|gbm', 'last')):
            q = M[M.task == task].iloc[0]; r('b', s, t, q.MAE_log, q.MAE_log_lo, q.MAE_log_hi, q.n, src='4A_metrics.csv')
    P = pd.read_csv(RES / '4A_paired.csv')
    for lab in ('target T3: origin T1 - origin T0', 'target T3: origin T2 - origin T0'):
        q = P[P.comparison == lab].iloc[0]; r('b2', 'seq', lab, q.dMAE_log, q.lo, q.hi, q.n, src='4A_paired.csv')
    for task, o in (('T0->T3|ftvhist|gbm', 'T0'), ('T1->T3|ftvhist|gbm', 'T1'), ('T2->T3|ftvhist|gbm', 'T2'), ('T0->T3|rollout|gbm', 'T0 rollout')):
        q = M[M.task == task].iloc[0]; r('b2', 'origin', o, q.MAE_log, q.MAE_log_lo, q.MAE_log_hi, q.n, src='4A_metrics.csv')
    L = pd.read_csv(R29 / 'landmark_metrics.csv')
    for rep in ('B', 'S'):
        for t in T2:
            q = L[(L.representation == rep) & (L.landmark == t)].iloc[0]; r('c', rep, t, q.auroc, q.auroc_lo, q.auroc_hi, q.n, src='forecasting landmark_metrics.csv')
    Tj = pd.read_csv(RES / '4B_trajectories.csv')
    for q in Tj.itertuples():
        r('c2', f'pcr{q.pcr}', q.landmark, q['mean'] if False else q[3], q.lo, q.hi, q.n, src='4B_trajectories.csv')
    for t in TR:
        for kind in ('ridge', 'gbm'):
            q = P[P.comparison == f'{t} {"Ridge" if kind == "ridge" else "GBM"}: state+arm - state'].iloc[0]
            r('d', kind, t, q.dMAE_log, q.lo, q.hi, q.n, src='4A_paired.csv')
    PC = pd.read_csv(R29 / 'paired_comparisons.csv')
    for t in T2:
        q = PC[PC.comparison == f'arm context at {t}'].iloc[0]; r('d2', 'pcr', t, q.d_auroc, q.lo, q.hi, src='forecasting paired_comparisons.csv')
    S = pd.read_csv(RES / '4D_summary.csv')
    s16 = S[(S.scenario == 'revision') & (S.factor == 1.6)]
    for s in SYS:
        g = s16[s16.system == s].set_index('rid')
        fw = g.loc[['F2a', 'F2b', 'F3']]; iss = g.loc[['R1a', 'R1b']]
        r('e', s, 'forward recovered', np.average(fw.recovered, weights=fw.n), n=int(fw.n.sum()), src='4D_summary.csv')
        r('e', s, 'issued recovered', np.average(iss.recovered, weights=iss.n), n=int(iss.n.sum()), src='4D_summary.csv')
        r('e', s, 'stale input', np.average(fw.stale_input_rate, weights=fw.n), src='4D_summary.csv')
        r('e', s, 'R1a MAE_log', g.at['R1a', 'MAE_log'], src='4D_summary.csv')
    ref = s16[(s16.system == 'reference') & (s16.rid == 'R1a')].iloc[0]; r('e', 'reference', 'R1a MAE_log', ref.MAE_log, src='4D_summary.csv')
    for f, g in S[(S.scenario == 'revision') & (S.system == 'uncorrected') & (S.rid.isin(['F2a', 'F2b', 'F3']))].groupby('factor'):
        r('e2', 'uncorrected', 'forward recovered', np.average(g.recovered, weights=g.n), x=f, src='4D_summary.csv')
    late = S[S.scenario == 'late']
    for s in ('oncostate', 'full_history', 'memory'):
        g = late[(late.system == s) & (late.rid.isin(['F2a', 'F2b', 'F3']))]
        r('e3', s, 'late forward recovered', np.average(g.recovered, weights=g.n), n=int(g.n.sum()), src='4D_summary.csv')
    Cst = pd.read_csv(RES / '4D_cost.csv').groupby('system').mean(numeric_only=True)
    for s in ('oncostate', 'full_history'):
        r('e4', s, 'records', Cst.at[s, 'records_reread']); r('e4', s, 'model runs', Cst.at[s, 'model_runs']); r('e4', s, 'ms', 1000 * Cst.at[s, 'sec'])
    TA = pd.read_csv(RES / '4E_tace_reproduction.csv').set_index('system')
    for s, lab in (('naive: train_mean', 'training mean'), ('enet WM-3', 'Elastic Net'), ('gbr WM-3', 'GBM (state)')):
        r('f', 'tace', lab, TA.at[s, 'MAE'], TA.at[s, 'MAE_lo'], TA.at[s, 'MAE_hi'], TA.at[s, 'n'], src='4E_tace_reproduction.csv')
    EX = pd.read_csv(RES / '4E_ispy1_ld.csv')
    for q in EX[EX.model.isin(['locf', 'gbm'])].itertuples():
        r('f2', f'{q.cohort}|{q.model}', q.transition, q.MAE_log, q.lo, q.hi, q.n, src='4E_ispy1_ld.csv')
    for t in TR:
        for lab, key in (('GBM: state - last', f'{t} GBM: state - last (representation)'), ('Ridge: state - last', f'{t} Ridge: state - last (representation)'),
                         ('state: GBM - Ridge', f'{t} state: GBM - Ridge (model)')):
            q = P[P.comparison == key].iloc[0]; r('s1', lab, t, q.dMAE_log, q.lo, q.hi, q.n, src='4A_paired.csv')
    for o, t in (('T0', 'T2'), ('T0', 'T3'), ('T1', 'T3')):
        for task, s in ((f'{o}->{t}|ftvhist|gbm', 'direct'), (f'{o}->{t}|rollout|gbm', 'rollout')):
            q = M[M.task == task].iloc[0]; r('s2', s, f'{o}->{t}', q.MAE_log, q.MAE_log_lo, q.MAE_log_hi, q.n, src='4A_metrics.csv')
    # ------------------------------------------------ richer main figure: example patient, scatter, calibration, trajectories, matrix, ECDF
    EX = '250274'                                      # representative I-SPY2 patient
    F28 = REPO / 'outputs/nature_figures/figure28/figures/assets'
    A = pd.read_csv(RES / 'audit_patient_table.csv', dtype={'patient_id': str}).set_index('patient_id')
    for t in T2:
        r('x_obs', t, str(F28 / f'representative_{t}.png'), A.at[EX, f'{t}_ftv_pe'], x=A.at[EX, f'{t}_day'], src='audit_patient_table.csv')
    r('x_info', 'pcr', 'final', A.at[EX, 'pcr']); r('x_info', 'arm', str(A.at[EX, 'arm']), np.nan)
    O = pd.read_csv(RES / '4A_oof.csv', dtype={'patient_id': str})
    for (o, t) in (('T0', 'T1'), ('T1', 'T2'), ('T2', 'T3')):
        q = O[(O.patient_id == EX) & (O.task == f'{o}->{t}|state|gbm')].iloc[0]; l = O[(O.patient_id == EX) & (O.task == f'{o}->{t}|last|locf')].iloc[0]
        r('x_fc', 'gbm', f'{o}->{t}', q.pred, q.pred - q.q90, q.pred + q.q90, src='4A_oof.csv'); r('x_fc', 'locf', f'{o}->{t}', l.pred, src='4A_oof.csv')
    R29o = pd.read_csv(R29 / 'oof_predictions.csv', dtype={'patient_id': str}); R29o = R29o[(R29o.scenario == 'natural') & (R29o.representation == 'S')]
    for q in R29o[R29o.patient_id == EX].itertuples():
        r('x_pcr', 'S', q.landmark, q.p, src='forecasting oof_predictions.csv')
    for q in O[O.task.isin([f'{t}|state|gbm' for t in TR])].itertuples():
        r('hex', q.task.split('|')[0], q.patient_id, q.pred, x=q.y, src='4A_oof.csv')
    CB = pd.read_csv(RES / '4B_calibration_bins.csv')
    for q in CB.itertuples():
        r('cal', q.landmark, str(q.bin), q.observed, x=q.mean_pred, n=q.n, src='4B_calibration_bins.csv')
    Ls = pd.read_csv(R29 / 'landmark_metrics.csv')
    for t in T2:
        q = Ls[(Ls.representation == 'S') & (Ls.landmark == t)].iloc[0]; r('cal_stat', t, 'slope', q.cal_slope, x=q.cal_intercept, src='forecasting landmark_metrics.csv')
    pv = R29o.pivot(index='patient_id', columns='landmark', values='p')[T2]; yy = R29o.groupby('patient_id').pcr.first()
    rng = np.random.default_rng(20261011); samp = rng.choice(pv.index, 160, replace=False)
    for pid in samp:
        for t in T2:
            r('spag', f'pcr{int(yy[pid])}|{pid}', t, pv.at[pid, t], x=T2.index(t), src='forecasting oof_predictions.csv (random sample of 160, seed 20261011)')
    DP = pd.read_csv(RES / '4D_predictions.csv', dtype={'patient_id': str})
    ex = DP[(DP.patient_id == EX) & (DP.scenario == 'revision') & (DP.factor == 1.6)]
    for q in ex.itertuples():
        r('x_rev', q.system, q.rid, q.pred, x=q.pre_correction, n=q.ref, src='4D_predictions.csv')
    S4 = pd.read_csv(RES / '4D_summary.csv'); S4 = S4[(S4.scenario == 'revision') & (S4.factor == 1.6)]
    for q in S4.itertuples():
        r('mat', q.system, q.rid, q.recovered, q.recovered_lo, q.recovered_hi, q.n, src='4D_summary.csv')
    fw = DP[(DP.scenario == 'revision') & (DP.factor == 1.6) & (DP.rid.isin(['F2b', 'F3']))].copy(); fw['dev'] = (fw.pred - fw.ref).abs()
    for s in ('uncorrected', 'memory', 'oncostate', 'full_history'):
        for v in np.sort(fw[fw.system == s].dev.dropna().to_numpy()):
            r('ecdf', s, 'pcr_dev', v, src='4D_predictions.csv (pCR at T2 and T3)')
    D = pd.DataFrame(R); D.to_csv(FIG / 'figure5_data.csv', index=False); return D


def main():
    d = build()
    print(f'wrote {FIG / "figure5_data.csv"} ({len(d)} rows)')


if __name__ == '__main__':
    main()
