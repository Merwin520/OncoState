#!/usr/bin/env python3
"""Part B endpoints (analysis plan, section 3) from results/revision_{events.csv.gz,patients.csv} -> tables/B_*.csv,
results/revision_summary.json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dv.common import CFG, S, best, evidence, prepare  # noqa: E402
from dv.revision import satisfies  # noqa: E402
from dv.stats import boot_idx, paired  # noqa: E402

T = HERE / 'tables'
B, SEED = CFG['bootstrap']['n'], CFG['bootstrap']['seed']
PROVED, INC = S.PROVED_EQUIVALENT, S.INCONCLUSIVE_INCOMPLETE_SET


def main():
    E = pd.read_csv(HERE / 'results/revision_events.csv.gz', dtype={'patient_id': str}, keep_default_na=False)
    Pt = pd.read_csv(HERE / 'results/revision_patients.csv', dtype={'patient_id': str}, keep_default_na=False)
    ev = evidence()
    x0 = {p: best(prepare(ev[p], 30.0))[0] for p in E.patient_id.unique()}
    a1 = E[E.arm == 'A1'].copy()
    a1['is_revision'] = [int(k == 'add' and not satisfies(x0[p], tuple(f.split(' ')))) if k == 'add' else int(k == 'retract')
                         for p, k, f in zip(a1.patient_id, a1.kind, a1.fact)]
    key = ['method', 'scenario', 'seed', 'patient_id']
    init = a1[a1.kind == 'init'].set_index(key)[['fresh_status', 'fresh_response']].rename(columns=lambda c: 'init_' + c[6:])
    a1 = a1.join(init, on=key)
    a1['invalidates'] = ((a1.kind != 'init') & ((a1.fresh_status != a1.init_status) | (a1.fresh_response != a1.init_response))).astype(int)
    per = a1.groupby(key).agg(n_events=('step', 'max'), n_revisions=('is_revision', 'sum'), any_invalidation=('invalidates', 'max'),
                              any_refuted=('fresh_status', lambda s: int((s == INC).any())), any_conflict=('fresh_status', lambda s: int((s == 'EVIDENCE_CONFLICT').any())))
    fin = E.loc[E.groupby(key + ['arm']).step.idxmax()]
    fin_w = fin.pivot_table(index=key, columns='arm', values=['accepted', 'stale_accepted', 'wrong', 'accepted_wrong'], aggfunc='first')
    fin_w.columns = [f'final_{v}_{a}' for v, a in fin_w.columns]
    anys = E.groupby(key + ['arm']).stale_accepted.max().unstack('arm').add_prefix('any_stale_')
    per = per.join(fin_w).join(anys)
    f1 = fin[fin.arm == 'A1'].set_index(key)
    per['final_status_A1'] = f1.status; per['final_response_A1'] = f1.response
    per = per.join(init).join(Pt.set_index(key)[['a1_calls', 'a2_calls', 'a1_seconds', 'final_reanchor', 'ref_response', 'stratum', 'role']])
    per = per.reset_index(); per.to_csv(T / 'B0_per_patient.csv', index=False)

    # ---------------------------------------------------------------- B1: real R03 revisions (descriptive)
    out = {}
    for m in ('C0', 'CA'):
        r = per[(per.scenario == 'R-real') & (per.method == m)]
        if not len(r):
            continue
        out[f'B1_{m}'] = dict(n=len(r), with_revision=int((r.n_revisions > 0).sum()), invalidated=int(r.any_invalidation.sum()),
                              refuted_set=int(r.any_refuted.sum()), conflict=int(r.any_conflict.sum()),
                              init_proved=int((r.init_status == PROVED).sum()),
                              init_proved_then_invalidated=int(((r.init_status == PROVED) & (r.any_invalidation == 1)).sum()),
                              init_proved_then_refuted=int(((r.init_status == PROVED) & (r.any_refuted == 1)).sum()),
                              final_status=r.final_status_A1.value_counts().to_dict(),
                              init_wrong=int((r.init_response != r.ref_response).sum()), final_wrong_A1=int(r.final_wrong_A1.sum()),
                              final_reanchor=r[r.final_reanchor != ''].final_reanchor.value_counts().to_dict())
        pd.crosstab(r.init_status, r.final_status_A1).to_csv(T / f'B1_R03_status_transition_{m}.csv')

    # ---------------------------------------------------------------- B2-B4 by scenario (R-sim: patient mean over seeds)
    rows = []; contrasts = []
    for m in sorted(per.method.unique()):
        for sc in ('R-real', 'R-sim', 'R-err'):
            r = per[(per.scenario == sc) & (per.method == m)]
            if not len(r):
                continue
            g = r.groupby('patient_id').mean(numeric_only=True)
            n = len(g)
            for arm in ('A1', 'A2', 'A3', 'A4', 'A5'):
                rows.append(dict(method=m, scenario=sc, arm=arm, n_patients=n, n_replays=len(r),
                                 any_stale_accepted=g[f'any_stale_{arm}'].mean(), final_stale_accepted=g[f'final_stale_accepted_{arm}'].mean(),
                                 final_accepted=g[f'final_accepted_{arm}'].mean(), final_wrong=g[f'final_wrong_{arm}'].mean(),
                                 final_accepted_wrong=g[f'final_accepted_wrong_{arm}'].mean()))
            idx = boot_idx(n, B, SEED)
            for lhs, rhs, col in (('A3', 'A2', 'any_stale_'), ('A4', 'A2', 'any_stale_'), ('A3', 'A2', 'final_accepted_wrong_'),
                                  ('A5', 'A2', 'final_accepted_wrong_'), ('A4', 'A2', 'final_wrong_')):
                x = g[col + lhs].to_numpy(float) - g[col + rhs].to_numpy(float)
                pt, ci, p = paired(lambda ix: x[ix].mean(), idx)
                contrasts.append(dict(method=m, scenario=sc, contrast=f'{lhs} - {rhs}', endpoint=col.strip('_'), diff=pt, lo=ci[0], hi=ci[1], p=p))
            eq = E[(E.method == m) & (E.scenario == sc) & (E.arm == 'A2')]
            rows[-1].update()
            out[f'B4_{m}_{sc}'] = dict(events=len(eq), a2_mismatch_events=int((eq.a2_equals_a1 == 0).sum()),
                                       a1_calls=int(r.a1_calls.sum()), a2_calls=int(r.a2_calls.sum()), call_ratio=float(r.a2_calls.sum() / r.a1_calls.sum()),
                                       a1_solver_s=float(r.a1_seconds.sum()))
    B2 = pd.DataFrame(rows); B2.to_csv(T / 'B2_arms_by_scenario.csv', index=False)
    BC = pd.DataFrame(contrasts); BC.to_csv(T / 'B3_paired_contrasts.csv', index=False)
    reanc = per[per.final_reanchor != ''].groupby(['method', 'scenario']).final_reanchor.value_counts().rename('n').reset_index()
    reanc.to_csv(T / 'B5_reanchor.csv', index=False)
    (HERE / 'results/revision_summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)); print(B2.round(3).to_string()); print(BC.round(4).to_string()); print(reanc.to_string())


if __name__ == '__main__':
    main()
