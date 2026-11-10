#!/usr/bin/env python3
"""Three-reader table: readers R02, R03 and R04 and the majority consensus (partner chosen by >= 2 of 3 readers, otherwise the
V0 partner), against the adjudicated reference. Patient-level response and F1 use the comments reading (merge statements count as corrections).
Writes results/readers3_rows.csv, readers3_patients.csv, readers3_summary.json.
    python readers3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_readers as RR  # noqa: E402

OUT = RR.OUT
SEED = 20261025


def etype(v0, ref):
    if v0 == ref:
        return 'correct'
    if v0 == '':
        return 'missed match'          # V0 called the lesion gone, reference has a partner
    if ref == '':
        return 'false match'           # V0 paired a lesion the reference calls gone
    return 'wrong partner'


def main():
    C = pd.read_csv(OUT / 'correspondence_level.csv', dtype=str, keep_default_na=False)
    for k in ('v0_correct', 'reader_correct', 'changed'):
        C[k] = C[k].astype(int)
    C['etype'] = [etype(a, b) for a, b in zip(C.v0, C.ref)]
    w = C.pivot_table(index=['patient_id', 'b', 'v0', 'ref', 'etype'], columns='reader', values='reader_partner', aggfunc='first').reset_index()
    rates = {}

    def maj(row):
        votes = pd.Series([row.R02, row.R03, row.R04]).value_counts()
        return votes.index[0] if votes.iloc[0] >= 2 else row.v0
    w['consensus'] = w.apply(maj, axis=1)
    w.to_csv(OUT / 'readers3_rows.csv', index=False)

    # patient level: responses, F1, time
    ev = RR.evidence(); V = RR.Vault()
    adj = json.loads((RR.FF / 'adjudication.json').read_text())['patients']
    P0 = pd.read_csv(OUT / 'patient_level.csv', dtype={'patient_id': str})
    rows = []
    for p, g in w.groupby('patient_id'):
        P = RR.prepare(ev[p], 30.0)
        ref = sorted((b, f) for b, f in zip(g.b, g.ref) if f) if p in adj else V.pairs(p)
        v0 = sorted(P.v0_repository)
        extra_ref = [x for x in ref if x[0] not in set(g.b)]
        rec = dict(patient_id=p, resp_ref=RR.response_cc(P.bl_t, P.fu_t, ref), resp_v0=RR.response_cc(P.bl_t, P.fu_t, v0),
                   f1_v0=RR.f1(v0, ref))
        for r in ('R02', 'R03', 'R04', 'consensus'):
            pairs = sorted({(b, f) for b, f in zip(g.b, g[r]) if f})
            rec[f'resp_{r}'] = RR.response_cc(P.bl_t, P.fu_t, pairs)
            rec[f'f1_{r}'] = RR.f1(pairs, ref)
        q = P0[P0.patient_id == p].set_index('reader')
        assert (q.resp_ref == rec['resp_ref']).all(), p
        for r in ('R02', 'R03', 'R04'):                            # real readers: free-text merge/split comments count as corrections
            rec[f'resp_{r}'], rec[f'f1_{r}'] = q.at[r, 'resp_comments'], q.at[r, 'f1_comments']
        t = q.seconds
        rec.update(sec_R02=t['R02'], sec_R03=t['R03'], sec_R04=t['R04'],
                   n_edits_R02=int((g.R02 != g.v0).sum()), n_edits_R03=int((g.R03 != g.v0).sum()), n_edits_R04=int((g.R04 != g.v0).sum()),
                   n_v0_errors=int((g.etype != 'correct').sum()), extra_ref_pairs=len(extra_ref))
        rows.append(rec)
    T = pd.DataFrame(rows); T.to_csv(OUT / 'readers3_patients.csv', index=False)
    out = dict()
    for r in ('R02', 'R03', 'R04', 'consensus'):
        e = w[w.etype != 'correct']; ok = w[w.etype == 'correct']
        w0 = T.resp_v0 != T.resp_ref; w1 = T[f'resp_{r}'] != T.resp_ref
        out[r] = dict(errors_fixed=int((e[r] == e.ref).sum()), errors=len(e), errors_changed=int((e[r] != e.v0).sum()),
                      correct_broken=int((ok[r] != ok.ref).sum()), correct=len(ok),
                      by_type={t: [int((g[r] == g.ref).sum()), len(g)] for t, g in e.groupby('etype')},
                      f1=[T.f1_v0.mean(), T[f'f1_{r}'].mean()],
                      response=dict(v0_wrong=int(w0.sum()), wrong_after=int(w1.sum()), corrected=int((w0 & ~w1).sum()),
                                    newly_wrong=int((~w0 & w1).sum()), changed=int((T[f'resp_{r}'] != T.resp_v0).sum())))
    (OUT / 'readers3_summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
