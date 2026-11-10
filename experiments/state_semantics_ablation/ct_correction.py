#!/usr/bin/env python3
"""Ablation F on the CT benchmark (Exp 2.2): the same 60 real matching-error correction events (exp22b_retention.events, seed 20260921).

A Full                     official procedure: correction written to the persistent state (relink), state saved, re-loaded, Q1-Q3 answered
F Non-durable correction   correction applied to the result returned in the review session only (Q1); the persistent state is not updated,
                           so later sessions (Q2, Q3) answer from the stored, uncorrected state.
B, D, E: N/A on CT (the Exp 2.2 CT state has no version store, no missingness semantics and no field-level provenance to remove).
Full must reproduce outputs/exp22_external_final/edit_retention.csv exactly, otherwise the script stops."""
import json
import pickle
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E1 = REPO / 'experiments/exp1_agentic_longitudinal'
sys.path.insert(0, str(E1))
import exp22b_common as C  # noqa: E402
import exp22b_retention as RET  # noqa: E402

OUT = REPO / 'outputs/state_semantics_ablation'


def answers(rows, e):
    pred = C.pred_from_rows(rows, e['bl'], e['fu'])
    return {'Q1': (e['pb'], e['pf']) in pred['pairs'],
            'Q2': pred['status_b'].get(e['pb']) == 'persistent' and pred['status_f'].get(e['pf']) == 'persistent',
            'Q3': (e['pb'], e['pf']) in pred['change']}


def persist_reload(rows, e):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / 'state.json'
        f.write_text(json.dumps({'correspondences': rows, 'baseline_lesions': {str(k): v for k, v in e['bl'].items()},
                                 'followup_lesions': {str(k): v for k, v in e['fu'].items()}}))
        st = json.loads(f.read_text())
    return st['correspondences'], {int(k): v for k, v in st['baseline_lesions'].items()}, {int(k): v for k, v in st['followup_lesions'].items()}


def main():
    det, _ = pickle.loads((REPO / 'outputs/stateful_baseline/det_cache.pkl').read_bytes())
    evs, n_all = RET.events(det)
    rows = []
    for e in evs:
        edited = RET.relink(e['rows'], e['pb'], e['pf'])
        # A Full: corrected state persisted and re-loaded for every later query
        r, bl, fu = persist_reload(edited, e)
        full = answers(r, dict(e, bl=bl, fu=fu))
        # F: review session sees the corrected result; the stored state keeps the uncorrected correspondences
        r0, bl0, fu0 = persist_reload(e['rows'], e)
        stored = answers(r0, dict(e, bl=bl0, fu=fu0))
        view = answers(edited, e)
        nondur = {'Q1': view['Q1'], 'Q2': stored['Q2'], 'Q3': stored['Q3']}
        for q in ('Q1', 'Q2', 'Q3'):
            rows.append(dict(variant='A_full', protocol='CT_correction_60', patient_id=e['pid'], pb=e['pb'], pf=e['pf'], query=q,
                             reflects_correction=bool(full[q]), context='persistent state re-loaded'))
            rows.append(dict(variant='F_nondurable_correction', protocol='CT_correction_60', patient_id=e['pid'], pb=e['pb'], pf=e['pf'], query=q,
                             reflects_correction=bool(nondur[q]),
                             context='review session (corrected result returned)' if q == 'Q1' else 'later session (stored state uncorrected)'))
    df = pd.DataFrame(rows)
    arch = pd.read_csv(E1 / 'outputs/exp22_external_final/edit_retention.csv', dtype={'patient_id': str})
    arch = arch[arch.system == 'Full OncoAgent'][['patient_id', 'pb', 'pf', 'query', 'reflects_correction']].reset_index(drop=True)
    mine = df[df.variant == 'A_full'][['patient_id', 'pb', 'pf', 'query', 'reflects_correction']].reset_index(drop=True)
    same = arch.astype(str).equals(mine.astype(str))
    print(f'{len(evs)} events from {n_all} real matching errors; Full reproduces archived edit_retention.csv: {same}')
    if not same:
        sys.exit('STOP: Full CT correction retention does not reproduce the archive')
    df.to_csv(OUT / 'correction_retention_ct.csv', index=False)
    print(df.groupby(['variant', 'query']).reflects_correction.agg(['mean', 'sum', 'count']).to_string())


if __name__ == '__main__':
    main()
