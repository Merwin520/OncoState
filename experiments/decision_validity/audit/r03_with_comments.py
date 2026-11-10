#!/usr/bin/env python3
"""R03 correspondences with the reader's merge / split comments counted as corrections (every merge / split comment is a
correction). Rule-derived response with RECIST 1.1 coalescence handling:
lesions joined by correspondences form connected components; a component containing a target contributes the summed follow-up diameter
of its follow-up members once (merge: one diameter; split: sum of fragments). For one-to-one correspondences this equals the repository
engine response used everywhere else (asserted for V0, reference and strict R03 on all 50 patients).
    python audit/r03_with_comments.py   -> audit/r03_with_comments.csv, audit/r03_with_comments_summary.json"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from dv.common import Vault, evidence, prepare  # noqa: E402
from run_revision import r03_pairs  # noqa: E402
from cdtr.rules import MEASURABLE_TENTHS, engine_response, targets_of  # noqa: E402

B = lambda *n: {f'B{i}' for i in n}  # noqa: E731
F = lambda *n: {f'F{i}' for i in n}  # noqa: E731
GROUPS = {'73f7634ab3': [(B(12, 13), F(4))], '7eabe3a164': [(B(12, 17), F(2))], '2a452c50d7': [(B(12, 13, 14), F(7, 9))],   # = score_r03_comments.py
          'c9e1074f5b': [(B(22, 23, 24), F(7))], '2723d092b6': [(B(1, 2, 3, 4), F(1, 2))], '335f535208': [(B(7), F(5))],
          'a861405761': [(B(7), F(4, 5))], 'a0a080f42e': [(B(1, 2, 3), F(1, 2))]}


def response_cc(bl_t, fu_t, pairs):
    """Merge/split-aware rule inputs -> repository engine."""
    par = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for b, f in pairs:
        par[find(('b', b))] = find(('f', f))
    comp = {}
    for b, f in pairs:
        r = find(('b', b)); comp.setdefault(r, [set(), set()]); comp[r][0].add(b); comp[r][1].add(f)
    tg = targets_of(bl_t); tgs = set(tg)
    s0 = sum(bl_t[a] for a in tg)
    s1 = sum(sum(fu_t[f] for f in fs) for bs, fs in comp.values() if bs & tgs)
    mb, mf = {b for b, _ in pairs}, {f for _, f in pairs}
    new = any(d >= MEASURABLE_TENTHS for f, d in fu_t.items() if f not in mf)
    nts = [a for a in bl_t if a not in tgs]
    nt = 'ABSENT' if not nts else ('CR' if all(a not in mb for a in nts) else 'NON-CR/NON-PD')
    gone = bool(tg) and all(a not in mb for a in tg)
    case = dict(history=[s0 / 10] if tg else [], current_sum=s1 / 10 if tg else None, new_lesions=bool(new), non_target=nt,
                all_disappeared=gone, nodes_lt10=gone)
    return engine_response(case)['response']


def main():
    ev = evidence(); V = Vault(); r03 = r03_pairs()
    rows = []
    for p in sorted(r03):
        P = prepare(ev[p], 30.0)
        v0, ref, strict = sorted(P.v0_repository), V.pairs(p), r03[p]
        grp = {(b, f) for bs, fs in GROUPS.get(p, []) for b in bs for f in fs}
        assert all(b in P.bids and f in P.fids for b, f in grp), p
        comm = sorted(set(strict) | grp)
        for x in (v0, ref, strict):
            if len({b for b, _ in x}) == len(x) and len({f for _, f in x}) == len(x):
                assert response_cc(P.bl_t, P.fu_t, x) == P.response(x)[0], p
        y = {k: response_cc(P.bl_t, P.fu_t, x) for k, x in (('v0', v0), ('ref', ref), ('strict', strict), ('comments', comm))}
        rows.append(dict(patient_id=p, has_comment_group=int(p in GROUPS), v0=y['v0'], ref=y['ref'], r03_strict=y['strict'], r03_comments=y['comments'],
                         n_pairs_strict=len(strict), n_pairs_comments=len(comm)))
    D = pd.DataFrame(rows)
    out = {}
    for k in ('r03_strict', 'r03_comments'):
        w0, w1 = D.v0 != D.ref, D[k] != D.ref
        out[k] = dict(v0_wrong=int(w0.sum()), wrong_after=int(w1.sum()), response_changed=int((D[k] != D.v0).sum()),
                      corrected=int((w0 & ~w1).sum()), newly_wrong=int((~w0 & w1).sum()), still_wrong=int((w0 & w1).sum()))
        D[f'{k}_wrong'] = w1.astype(int)
    D['v0_wrong'] = (D.v0 != D.ref).astype(int)
    D.to_csv(HERE / 'r03_with_comments.csv', index=False)
    (HERE / 'r03_with_comments_summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(D[(D.v0_wrong == 1) | (D.r03_comments_wrong == 1) | (D.r03_strict_wrong == 1)].to_string())


if __name__ == '__main__':
    main()
