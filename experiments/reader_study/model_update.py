#!/usr/bin/env python3
"""Cross-patient matcher updating from the reader-study corrections.

The final correspondences of each reader (readers3_rows.csv: R02, R03, R04, all real; consensus = partner chosen by
>= 2 of 3 readers, otherwise V0) are turned into Exp 2.6 review events and passed, unchanged, through the Exp 2.6 package:
candidate repair (fit_candidate) -> validation gate on the 80 gate patients (run_gate) -> 144 unseen future patients.
Clinical endpoints on the future patients use the Exp 3.0 rule and paired bootstrap (experiments/exp30_correction_clinical_impact).
Reference correspondences are used only for gate and evaluation, never for fitting.

Events per review patient (one per baseline row, then one per follow-up lesion the reader left unmatched):
  partner == V0 partner   -> accept       partner != V0 partner -> correct      no partner -> disappeared (original = V0 partner)
  follow-up lesion without partner -> new_lesion (supervision only, not counted as a correction)
Writes real_readers/model_update_{matching,clinical,gate}.csv.
    python model_update.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'results'
sys.path.insert(0, str(REPO / 'experiments/exp30_correction_clinical_impact'))
import run_impact as X  # noqa: E402  (chdir into the Exp 2.6 package and puts it on the path)
from evaluation.clinician_coevolution import _future, _row  # noqa: E402

READERS = {'R02': 'R1', 'R03': 'R2', 'R04': 'R3', 'consensus': '2 of 3'}


def events_of(rows, cases, col):
    ev, t = [], 0
    for p, g in rows.groupby('patient_id'):
        partner = {r.b: getattr(r, col) for r in g.itertuples()}
        for r in g.itertuples():
            v, x = r.v0 or None, getattr(r, col) or None
            if x is not None and x == v:
                act, et = 'accept', 'lesion_match'
            elif x is not None:
                act, et = 'correct', 'lesion_match'
            else:
                act, et = 'disappeared', 'disappeared_lesion'
            t += 1
            ev.append(dict(patient_id=p, event_type=et, action=act, baseline_lesion_id=r.b, original_followup_lesion_id=v,
                           corrected_followup_lesion_id=x if act == 'correct' else None, timestamp=f'{t:08d}'))
        used = {x for x in partner.values() if x}
        for f in cases[p]['followup_lesions']:
            if f['lesion_id'] not in used:
                t += 1
                ev.append(dict(patient_id=p, event_type='new_lesion', action='new_lesion', baseline_lesion_id=None,
                               original_followup_lesion_id=f['lesion_id'], corrected_followup_lesion_id=None, timestamp=f'{t:08d}'))
    return ev


def main():
    cfg = X.load_config('configs/experiment_real.yaml'); seed, nb = cfg['random_seed'], cfg['bootstrap_iterations']
    bundle = X.load_bundle(X.SOURCE); cases, gt = bundle['cases'], X.load_gt(X.SOURCE)
    batches = X.make_batches(bundle['manifest'], json.loads(X.ALL_REVIEW_PATIENTS.read_text(encoding='utf-8')))
    fut, gate_pids = batches['future_eval_1'], batches['validation_gate_1']
    rows = pd.read_csv(OUT / 'readers3_rows.csv', dtype=str, keep_default_na=False)
    assert not set(rows.patient_id) & (set(fut) | set(gate_pids))
    v0 = X.CentroidHungarianMatcher(cfg['v0']['gate_mm'])
    for p, g in rows.groupby('patient_id'):                       # reader table V0 == Exp 2.6 V0
        assert {(r.b, r.v0) for r in g.itertuples() if r.v0} == set(map(tuple, v0.predict(cases[p])['pairs'])), p
    p0, df0 = _future(v0, cases, gt, fut)
    match, clin, gates = [], X.outcome_rows(cases, gt, fut, p0, 'V0', 'clinician'), []
    match.append(_row('V0', [], df0, df0, p0, p0, cases, gt, fut, nb, seed, dict(readers='V0')))
    for col, lab in READERS.items():
        evs = events_of(rows, cases, col)
        n_corr = int(((rows[col] != rows.v0)).sum())
        cand = X.fit_candidate(evs, cases, cfg, seed)
        pc, dfc = _future(cand, cases, gt, fut)
        g = X.run_gate(cases, gt, v0, cand, gate_pids, cfg['non_regression_thresholds'], nb, seed)
        gates.append(dict(readers=lab, n_corrections=n_corr, accepted=g['accepted'], reason=g['reason'],
                          source='clinician'))
        pg, dfg = (pc, dfc) if g['accepted'] else (p0, df0)
        for cond, pp, dd in (('naive', pc, dfc), ('gated', pg, dfg)):
            match.append(_row(cond, evs, dd, df0, pp, p0, cases, gt, fut, nb, seed, dict(readers=lab, n_corrections=n_corr)))
            clin += X.outcome_rows(cases, gt, fut, pp, f'{cond}|{lab}', 'clinician')
    M = pd.DataFrame(match); C = pd.DataFrame(clin)
    pairs = []
    for cond in C.condition.unique():
        if cond == 'V0':
            continue
        ch, out = X.paired(C, cond, 'V0')
        pairs += [dict(condition=cond, endpoint=ep, changed_responses=ch, **v) for ep, v in out.items()]
    M.to_csv(OUT / 'model_update_matching.csv', index=False)
    pd.DataFrame(pairs).to_csv(OUT / 'model_update_clinical.csv', index=False)
    pd.DataFrame(gates).to_csv(OUT / 'model_update_gate.csv', index=False)
    pd.set_option('display.width', 200)
    print(pd.DataFrame(gates).to_string(index=False))
    print(M[['condition', 'readers', 'n_corrections', 'future_matching_f1', 'delta_f1_vs_v0', 'delta_f1_ci_lo', 'delta_f1_ci_hi',
             'future_errors_fixed', 'future_errors_introduced']].round(4).to_string(index=False))
    print(pd.DataFrame(pairs)[['condition', 'endpoint', 'base_rate', 'cond_rate', 'diff', 'lo', 'hi', 'mcnemar_p']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
