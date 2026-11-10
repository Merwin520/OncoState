#!/usr/bin/env python3
"""Ablation C (persistent lesion identity) on real multi-visit CT: HCUCH RECIST CT (public, 22 patients; 10 patient-region sequences with
>=3 visits, 21 RECIST target lesions whose cross-visit identity is given by the dataset's lesion alias = ground truth).

Shared by both conditions (unchanged Experiment 2.2 components): every segmented lesion instance of the dataset's reference masks,
rigid registration exp22_run.register, frozen hybrid matcher exp22_run.match_fast (30 mm gate, Hungarian, unmatched cost 0.4).
  C  pairwise only    : consecutive visits are matched (k-1 -> k); visit-local IDs + stored pairwise links; longitudinal identity is
                        reconstructed by chaining the links (a lesion with no link to the previous visit starts a new identity).
  Full canonical      : canonical lesion objects persist in the state; at visit k the new lesions are matched to the objects seen at k-1
                        exactly as in C, then lesions still unmatched are matched (same matcher, registration j -> k) against canonical
                        objects last observed at an earlier visit j < k-1 (re-identification across a gap).
                        NOTE: the frozen CT pipeline is two-visit; this canonical registry is a minimal multi-visit extension written
                        for this experiment.
Detector-miss sensitivity: each target-lesion instance at an intermediate visit is removed from the mask with probability q in
{0, 0.1, 0.2, 0.3} (same removals for both conditions; 20 seeds for q > 0). Endpoint per ground-truth target lesion: identity
preserved = all its retained instances carry one predicted identity that no other target lesion carries. Patient-level bootstrap.
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'experiments/exp1_agentic_longitudinal'))
sys.path.insert(0, str(REPO / 'experiments/exp28_repeated_synchronization'))
import exp22_run as R  # noqa: E402
from evaluation.statistics import paired_bootstrap, patient_bootstrap  # noqa: E402

DATA = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'HCUCH RECIST CT/final-formatted/final-formatted'))
OUT = REPO / 'outputs/state_semantics_ablation/lesion_identity'
ORDER = {'baseline': 0, 'follow-up-1': 1, 'follow-up-2': 2, 'follow-up-3': 3}
QS, SEEDS = [0.0, .1, .2, .3], range(20)


def sequences():
    r = pd.read_csv(DATA / 'recist_measurements.csv')
    r['k'] = r.study_order.map(ORDER)
    r['base'] = r.patient_id.map('{:03d}'.format) + '_' + r.uuid
    seq = r.groupby(['patient_id', 'region']).k.nunique()
    out = []
    for (pid, reg), _ in seq[seq >= 3].items():
        d = r[(r.patient_id == pid) & (r.region == reg)].sort_values('k')
        visits = []
        for k, g in d.groupby('k'):
            base = g.base.iloc[0]
            img = next(DATA.glob(f'images/{reg}/*/images/{base}.nii.gz')); msk = next(DATA.glob(f'images/{reg}/*/masks/{base}.nii.gz'))
            visits.append(dict(k=int(k), img=str(img), mask=str(msk), targets={str(a): int(v) for a, v in zip(g.lesion_label_alias, g.lesion_label_value)}))
        out.append(dict(patient_id=int(pid), region=reg, visits=visits))
    return out


def tx_for(s, i, j):
    return R.register(s['visits'][i]['img'], s['visits'][j]['img'], OUT / f"reg/{s['patient_id']}_{s['region']}_{i}_{j}.tfm")


def full_cost(s, i, j):
    """Frozen match_fast cost terms for ALL lesion pairs of visits i, j (copied term-for-term from exp22_run.match_fast; the cost of a
    pair does not depend on which other lesions are present, so the assignment on any subset equals match_fast on the masked images —
    verified against match_fast in check_equivalence())."""
    from scipy import ndimage
    bl_img, fu_img = sitk.ReadImage(s['visits'][i]['mask']), sitk.ReadImage(s['visits'][j]['mask'])
    tx = tx_for(s, i, j)
    b_arr, bd = R.describe(bl_img); f_arr, fd = R.describe(fu_img)
    bids, fids = sorted(bd), sorted(fd)
    C = np.full((len(bids), len(fids)), 100.)
    if bd and fd:
        warped = sitk.GetArrayFromImage(sitk.Resample(bl_img, fu_img, tx.GetInverse(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt32))
        wobjs = ndimage.find_objects(warped.astype(np.int32))
        for a_, bi in enumerate(bids):
            bc = np.array(tx.TransformPoint(tuple(bd[bi]['centroid_mm'])))
            for c_, fj in enumerate(fids):
                d = float(np.linalg.norm(bc - np.array(fd[fj]['centroid_mm'])))
                if d > 30.:
                    continue
                sw = wobjs[bi - 1] if bi - 1 < len(wobjs) else None
                fb = fd[fj]['bbox']
                if sw is None:
                    iou = 0.
                else:
                    lo = [min(sw[x].start, fb[x][0]) for x in range(3)]; hi = [max(sw[x].stop, fb[x][1]) for x in range(3)]
                    sl = tuple(slice(lo[x], hi[x]) for x in range(3))
                    bm = warped[sl] == bi; fm = f_arr[sl] == fj
                    u = np.count_nonzero(bm | fm)
                    iou = float(np.count_nonzero(bm & fm) / u) if u else 0.
                bv, fv = bd[bi]['volume_mm3'], fd[fj]['volume_mm3']
                C[a_, c_] = .45 * min(d / 30., 1.) + .30 * (1. - iou) + .15 * abs(bv - fv) / max(bv, fv)
    return bids, fids, C


def assign(cm, keep_i, keep_j):
    """match_fast assignment (unmatched cost 0.4, accept < 0.8) restricted to the kept lesions."""
    from scipy.optimize import linear_sum_assignment
    bids, fids, C = cm
    bi = [k for k, x in enumerate(bids) if x in keep_i]; fj = [k for k, x in enumerate(fids) if x in keep_j]
    n, m = len(bi), len(fj)
    cost = np.full((n + m, n + m), 100.)
    if n and m:
        cost[:n, :m] = C[np.ix_(bi, fj)]
    cost[:n, m:] = .4; cost[n:, :m] = .4; cost[n:, m:] = 0
    rr, cc = linear_sum_assignment(cost)
    return [(bids[bi[r]], fids[fj[c]]) for r, c in zip(rr, cc) if r < n and c < m and cost[r, c] < .8]


def check_equivalence(s, costs, rng):
    """Assignment on precomputed costs == frozen match_fast on (masked) images: full masks and 2 random subsets per visit pair."""
    def masked(path, keep):
        img = sitk.ReadImage(path); a = sitk.GetArrayFromImage(img)
        a = np.where(np.isin(a, list(keep)), a, 0).astype(np.uint32); o = sitk.GetImageFromArray(a); o.CopyInformation(img); return o
    bad = 0
    for (i, j), cm in costs.items():
        for t in range(3):
            ki = set(cm[0]) if t == 0 else {x for x in cm[0] if rng.random() < .7}
            kj = set(cm[1]) if t == 0 else {x for x in cm[1] if rng.random() < .7}
            rows, _, _ = R.match_fast(masked(s['visits'][i]['mask'], ki), masked(s['visits'][j]['mask'], kj), tx_for(s, i, j))
            ref = sorted((r['baseline_lesion_id'], r['followup_lesion_id']) for r in rows if r['status'] == 'persistent')
            bad += ref != sorted(assign(cm, ki, kj))
    return bad


def ids_present(s, k, drops, costs):
    ids = next(cm[0] for (i, j), cm in costs.items() if i == k) if any(i == k for i, j in costs) else \
        next(cm[1] for (i, j), cm in costs.items() if j == k)
    return set(ids) - set(drops.get(k, set()))


def tracks(s, drops, canonical, costs):
    """Return {(visit, lesion id): track id}."""
    n = len(s['visits']); tid = {}; nxt = 0; last_seen = {}
    for lid in sorted(ids_present(s, 0, drops, costs)):
        tid[(0, lid)] = nxt; last_seen[nxt] = (0, lid); nxt += 1
    for k in range(1, n):
        present = ids_present(s, k, drops, costs)
        for a, b in assign(costs[(k - 1, k)], ids_present(s, k - 1, drops, costs), present):
            t = tid[(k - 1, a)]; tid[(k, b)] = t; last_seen[t] = (k, b)
        if canonical:                                          # re-identify unmatched lesions against objects last seen before k-1
            for j in range(k - 2, -1, -1):
                un = present - {b for (kk, b) in tid if kk == k}
                cand = {lid for t, (kk, lid) in last_seen.items() if kk == j}
                if un and cand:
                    for a, b in assign(costs[(j, k)], cand, un):
                        t = tid[(j, a)]; tid[(k, b)] = t; last_seen[t] = (k, b)
        for b in sorted(present):
            if (k, b) not in tid:
                tid[(k, b)] = nxt; last_seen[nxt] = (k, b); nxt += 1
    return tid


def score(s, tid, drops):
    rows = []
    alias = sorted({a for v in s['visits'] for a in v['targets']})
    owner = {}
    for a in alias:
        inst = [(k, v['targets'][a]) for k, v in enumerate(s['visits']) if a in v['targets'] and v['targets'][a] not in drops.get(k, set())]
        owner[a] = {tid.get(x) for x in inst}
        rows.append(dict(alias=a, n_instances=len(inst), n_visits_gt=sum(a in v['targets'] for v in s['visits'])))
    for r in rows:
        ids = owner[r['alias']]
        shared = any(ids & owner[b] for b in alias if b != r['alias'])
        r.update(fragments=len(ids), identity_preserved=float(len(ids) == 1 and not shared), merged_with_other=float(shared))
    return rows


def run_sequence(s):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    n = len(s['visits'])
    costs = {(i, j): full_cost(s, i, j) for j in range(n) for i in range(j)}
    bad = check_equivalence(s, costs, np.random.default_rng(s['patient_id']))
    res = []
    for q in QS:
        for seed in (SEEDS if q > 0 else [0]):
            rng = np.random.default_rng(1000 * seed + int(q * 100) + s['patient_id'])
            drops = {k: {v for v in s['visits'][k]['targets'].values() if rng.random() < q} for k in range(1, n - 1)}   # intermediate visits
            for cond, canon in (('C_pairwise_only', False), ('A_canonical_identity', True)):
                for r in score(s, tracks(s, drops, canon, costs), drops):
                    res.append(dict(patient_id=s['patient_id'], region=s['region'], n_visits=n, miss_rate=q, seed=seed, condition=cond,
                                    n_dropped=sum(len(v) for v in drops.values()), **r))
    print(f"done {s['patient_id']} {s['region']} equivalence_mismatches={bad}", flush=True)
    return res, dict(patient_id=s['patient_id'], region=s['region'], equivalence_checks=3 * len(costs), equivalence_mismatches=bad)


def main():
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = sequences()
    (OUT / 'sequences.json').write_text(json.dumps(seqs, indent=1))
    with Pool(10) as pool:
        out = pool.map(run_sequence, seqs)
    res = [r for rs, _ in out for r in rs]
    eq = pd.DataFrame([e for _, e in out]); eq.to_csv(OUT / 'matcher_equivalence_check.csv', index=False)
    if eq.equivalence_mismatches.sum():
        sys.exit('STOP: precomputed assignment differs from the frozen matcher')
    df = pd.DataFrame(res); df.to_csv(OUT / 'per_lesion_results.csv', index=False)
    # patient-level: mean over target lesions, then over seeds
    pp = df.groupby(['miss_rate', 'condition', 'patient_id']).identity_preserved.mean().reset_index()
    agg, pair_rows = [], []
    for q, d in pp.groupby('miss_rate'):
        w = d.pivot(index='patient_id', columns='condition', values='identity_preserved')
        for c in w.columns:
            b = patient_bootstrap(w[c].values)
            agg.append(dict(miss_rate=q, condition=c, estimate=b['estimate'], ci95_lo=b['ci_low'], ci95_hi=b['ci_high'], n_patients=b['n_patients'],
                            n_target_lesions=int(df[(df.miss_rate == q) & (df.condition == c) & (df.seed == df.seed.min())].shape[0])))
        pb = paired_bootstrap(w['A_canonical_identity'].values, w['C_pairwise_only'].values)
        pair_rows.append(dict(miss_rate=q, comparison='A_canonical_identity - C_pairwise_only', difference=pb['estimate'], ci95_lo=pb['ci_low'],
                              ci95_hi=pb['ci_high'], n_patients=pb['n_patients'], wilcoxon_p=pb.get('wilcoxon_p')))
    pd.DataFrame(agg).to_csv(OUT / 'aggregate.csv', index=False); pd.DataFrame(pair_rows).to_csv(OUT / 'paired_bootstrap.csv', index=False)
    print(pd.DataFrame(agg).round(3).to_string(index=False)); print(pd.DataFrame(pair_rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
