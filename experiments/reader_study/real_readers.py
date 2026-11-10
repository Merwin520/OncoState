#!/usr/bin/env python3
"""Clinician reader study (R02, R03, R04): correspondence-level and clinical-response results against the hidden reference.

Inputs : $READER_DATA/R0{2,3,4}_review_results_merged.csv (one row per V0 correspondence / unmatched follow-up lesion, 50 patients)
Reference, V0 and RECIST rules: the same code as experiments/decision_validity (Vault, prepare, merge/split-aware response_cc).
Two readings of each reader:
  strict   - only the explicit row decisions (pairs = prior_lesion rows with a follow-up partner)
  comments - plus the merge/split groups the reader wrote in free-text comments (groups listed below;
             R02 comments are listed but not converted, they are not correspondence statements)
'uncertain' rows keep the V0 partner (the reader did not change it).
Adjudication ($READER_DATA/adjudication.json): patients where a further clinician ruled the readers right replace the hidden reference by the
readers' (identical) correspondences. Results are written for the original reference (summary_original_reference.json) and the adjudicated one.
    python real_readers.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FF = Path(os.environ.get('READER_DATA', REPO / 'reader_data'))   # reader exports + adjudication.json
DV = REPO / 'experiments/decision_validity'
sys.path.insert(0, str(DV)); sys.path.insert(0, str(DV / 'audit'))
from dv.common import Vault, evidence, prepare  # noqa: E402
from r03_with_comments import GROUPS, response_cc  # noqa: E402

OUT = HERE / 'results'


def reader_pairs(d):
    pr = d[(d.row_type == 'prior_lesion') & (d.clinician_followup_lesion_id != '')]
    return {p: sorted(set(zip(g.baseline_lesion_id, g.clinician_followup_lesion_id))) for p, g in pr.groupby('patient_id')}


def f1(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    tp = len(a & b)
    return 2 * tp / (len(a) + len(b)) if (a or b) else 1.0


ADJ = True
# R04 (third reader): merge statements in free-text comments, converted like R03's
R04_GROUPS = {'73f7634ab3': [({'B3', 'B4', 'B6'}, {'F1'}), ({'B2', 'B12', 'B13'}, {'F4'})], 'b1d10e7baf': [({'B2'}, {'F4'})]}


def main():
    OUT.mkdir(exist_ok=True)
    ev = evidence(); V = Vault()
    R = {r: pd.read_csv(FF / f'{r}_review_results_merged.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig') for r in ('R02', 'R03', 'R04')}
    pids = sorted(set(R['R02'].patient_id))
    prep = {p: prepare(ev[p], 30.0) for p in pids}
    v0 = {p: sorted(prep[p].v0_repository) for p in pids}; ref = {p: V.pairs(p) for p in pids}
    adj = json.loads((FF / 'adjudication.json').read_text())['patients'] if ADJ else []
    for p in adj:
        a_, b_ = (reader_pairs(R[r]).get(p, []) for r in ('R02', 'R03'))
        assert a_ == b_, p
        ref[p] = a_

    # ---------------- correspondence level (per baseline lesion: partner or none)
    rows, pat, out = [], [], {}
    for r, d in R.items():
        rp = reader_pairs(d)
        pr = d[d.row_type == 'prior_lesion']
        for x in pr.itertuples():
            refp = dict(ref[x.patient_id]).get(x.baseline_lesion_id, '')
            v0p = x.V0_followup_lesion_id; rdp = x.clinician_followup_lesion_id
            rows.append(dict(reader=r, patient_id=x.patient_id, b=x.baseline_lesion_id, v0=v0p, reader_partner=rdp, ref=refp,
                             decision=x.clinician_decision, v0_correct=int(v0p == refp), reader_correct=int(rdp == refp),
                             changed=int(rdp != v0p), seconds=float(x.patient_review_seconds or 0)))
        for p in pids:
            P = prep[p]
            strict = rp.get(p, [])
            grp = {(b, f) for bs, fs in {'R03': GROUPS, 'R04': R04_GROUPS}.get(r, {}).get(p, []) for b in bs for f in fs}
            comm = sorted(set(strict) | grp)
            y = {k: response_cc(P.bl_t, P.fu_t, x) for k, x in (('v0', v0[p]), ('ref', ref[p]), ('strict', strict), ('comments', comm))}
            pat.append(dict(reader=r, patient_id=p, f1_v0=f1(v0[p], ref[p]), f1_strict=f1(strict, ref[p]), f1_comments=f1(comm, ref[p]),
                            resp_v0=y['v0'], resp_ref=y['ref'], resp_strict=y['strict'], resp_comments=y['comments'],
                            seconds=float(d[d.patient_id == p].patient_review_seconds.astype(float).iloc[0])))
    C = pd.DataFrame(rows); Pt = pd.DataFrame(pat)
    tag = '' if ADJ else '_original_reference'
    C.to_csv(OUT / f'correspondence_level{tag}.csv', index=False); Pt.to_csv(OUT / f'patient_level{tag}.csv', index=False)

    for r in R:
        c = C[C.reader == r]; p_ = Pt[Pt.reader == r]
        err, ok = c[c.v0_correct == 0], c[c.v0_correct == 1]
        res = dict(n_rows=len(c), n_v0_errors=len(err), n_v0_correct=len(ok),
                   errors_changed=int(err.changed.sum()), errors_corrected_to_ref=int(err.reader_correct.sum()),
                   correct_kept=int((ok.changed == 0).sum()), correct_broken=int((ok.reader_correct == 0).sum()),
                   decisions=c.decision.value_counts().to_dict(),
                   rows_correct_v0=int(c.v0_correct.sum()), rows_correct_reader=int(c.reader_correct.sum()),
                   f1_v0_mean=p_.f1_v0.mean(), f1_strict_mean=p_.f1_strict.mean(), f1_comments_mean=p_.f1_comments.mean(),
                   median_min=p_.seconds.median() / 60, iqr_min=[p_.seconds.quantile(.25) / 60, p_.seconds.quantile(.75) / 60],
                   total_min=p_.seconds.sum() / 60)
        for k in ('strict', 'comments'):
            w0 = p_.resp_v0 != p_.resp_ref; w1 = p_[f'resp_{k}'] != p_.resp_ref
            res[f'response_{k}'] = dict(v0_wrong=int(w0.sum()), wrong_after=int(w1.sum()), changed=int((p_[f'resp_{k}'] != p_.resp_v0).sum()),
                                        corrected=int((w0 & ~w1).sum()), newly_wrong=int((~w0 & w1).sum()), still_wrong=int((w0 & w1).sum()))
        out[r] = res

    # ---------------- agreement between the two readers (same rows)
    a = C[C.reader == 'R02'].set_index(['patient_id', 'b']); b = C[C.reader == 'R03'].set_index(['patient_id', 'b'])
    j = a.join(b, lsuffix='_2', rsuffix='_3', how='inner')
    po = float((j.changed_2 == j.changed_3).mean())
    p1, p2 = j.changed_2.mean(), j.changed_3.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2)
    out['agreement'] = dict(n_rows=len(j), same_partner=float((j.reader_partner_2 == j.reader_partner_3).mean()),
                            change_decision_agreement=po, kappa_change=(po - pe) / (1 - pe) if pe < 1 else np.nan,
                            both_flag_v0_error=int(((j.changed_2 == 1) & (j.changed_3 == 1) & (j.v0_correct_2 == 0)).sum()),
                            v0_errors=int((j.v0_correct_2 == 0).sum()),
                            either_flag_v0_error=int((((j.changed_2 == 1) | (j.changed_3 == 1)) & (j.v0_correct_2 == 0)).sum()))
    pa = Pt[Pt.reader == 'R02'].set_index('patient_id'); pb = Pt[Pt.reader == 'R03'].set_index('patient_id')
    out['agreement']['response_same_strict'] = int((pa.resp_strict == pb.resp_strict).sum())
    out['R02_comments'] = R['R02'][R['R02'].patient_comment != ''].groupby('patient_id').patient_comment.first().to_dict()
    out['adjudicated_patients'] = adj
    (OUT / f'summary{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    ADJ = False; main()
    ADJ = True; main()
