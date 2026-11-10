#!/usr/bin/env python3
"""Patient-level endpoints vs the reference response (repository engine on REFERENCE correspondences, same diameters, same detections).
  python evaluate.py --split policy_validation|test
writes tables/<split>_patient_level_results.csv, tables/<split>_metrics_with_95ci.csv, tables/<split>_paired_differences.csv
(test split also copied to tables/patient_level_results.csv and tables/metrics_with_95ci.csv)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cdtr import schema as S  # noqa: E402
from cdtr.data import load_evidence  # noqa: E402
from cdtr.energy import prepare  # noqa: E402
from cdtr.reference import ReferenceForSplit  # noqa: E402
from run_cdtr import cfg  # noqa: E402

RES, TAB = HERE / 'results', HERE / 'tables'
ENDPOINTS = ['wrong_response', 'false_pd', 'missed_pd', 'false_new', 'missed_new']


def lesion_f1(pred, ref):
    pred, ref = set(map(tuple, pred)), set(map(tuple, ref))
    tp = len(pred & ref)
    return tp, len(pred), len(ref)


def patient_rows(split, final):
    c = cfg(); ev = load_evidence(split); ref = ReferenceForSplit(split, final_evaluation=final)
    truth = {}
    for p, e in ev.items():
        P = prepare(e, c['energy']['gate_mm'])
        r, inp, _ = P.response(ref.pairs(p))
        truth[p] = dict(P=P, ref_response=r, ref_new=inp['new_lesions'], ref_pairs=ref.pairs(p))
    rows = []
    for f in sorted((RES / split).glob('*.json')):
        blob = json.loads(f.read_text())
        for seed, run in blob['runs'].items():
            for K, recs in run['checkpoints'].items():
                for p, r in recs.items():
                    t = truth[p]; P = t['P']
                    _, inp, _ = P.response([tuple(x) for x in r['pairs']])
                    tp, npred, nref = lesion_f1(r['pairs'], t['ref_pairs'])
                    rows.append(dict(arm=f.stem, policy=blob['policy'], alpha=blob['alpha'], q=blob['q'], seed=int(seed), K=int(K),
                                     budget_per_patient=int(K) / len(recs), patient_id=p, decision=r['decision'],
                                     reference=t['ref_response'], wrong_response=int(r['decision'] != t['ref_response']),
                                     false_pd=int(r['decision'] == 'PD' and t['ref_response'] != 'PD'),
                                     missed_pd=int(r['decision'] != 'PD' and t['ref_response'] == 'PD'),
                                     false_new=int(inp['new_lesions'] and not t['ref_new']), missed_new=int(t['ref_new'] and not inp['new_lesions']),
                                     tp=tp, n_pred=npred, n_ref=nref, n_reviews=r['n_reviews'], status=r.get('status'),
                                     reachable=';'.join(r.get('reachable', []) or []), reanchored=r.get('reanchored'),
                                     identity_certain=r.get('identity_certain'), split_sha256=blob['split_sha256'],
                                     config_sha256=blob['config_sha256']))
    return pd.DataFrame(rows), truth


def summarise(g):
    n = g.patient_id.nunique()
    d = {e: g[e].mean() for e in ENDPOINTS}
    tp, npd, nr = g.tp.sum(), g.n_pred.sum(), g.n_ref.sum()
    d['lesion_f1'] = 2 * tp / (npd + nr) if npd + nr else np.nan
    d['reviews'] = g.n_reviews.sum() / g.seed.nunique()
    errs = g.wrong_response.sum() / g.seed.nunique()
    d['wrong_per_100_reviews'] = 100 * errs / d['reviews'] if d['reviews'] else np.nan
    if g.status.notna().any():
        acc = g.status == S.PROVED_EQUIVALENT
        d['accepted_fraction'] = acc.mean()
        d['wrong_among_accepted'] = g.wrong_response[acc].mean() if acc.any() else np.nan
        d['deferred_fraction'] = 1 - acc.mean()
        d['label_coverage'] = np.mean([ref in reach.split(';') for ref, reach in zip(g.reference, g.reachable)])
    if g.identity_certain.notna().any():
        acc = g.identity_certain.astype(bool)
        d['accepted_fraction'] = acc.mean(); d['deferred_fraction'] = 1 - acc.mean()
        d['wrong_among_accepted'] = g.wrong_response[acc].mean() if acc.any() else np.nan
    d['n_patients'] = n
    return d


def boot(df, n_boot, seed):
    """Patient-level bootstrap (resample patients; seeds of the random arm averaged within patient) - marginal and paired vs cdtr."""
    rng = np.random.default_rng(seed)
    per = df.groupby(['arm', 'K', 'patient_id'])[ENDPOINTS + ['tp', 'n_pred', 'n_ref']].mean().reset_index()
    pids = np.array(sorted(per.patient_id.unique()))
    idx = rng.integers(0, len(pids), (n_boot, len(pids)))
    marg, paired = [], []
    for (arm, K), g in per.groupby(['arm', 'K']):
        g = g.set_index('patient_id').loc[pids]
        for e in ENDPOINTS:
            v = g[e].to_numpy(); bs = v[idx].mean(1)
            marg.append(dict(arm=arm, K=K, endpoint=e, value=v.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
        tp, npd, nr = (g[c_].to_numpy() for c_ in ('tp', 'n_pred', 'n_ref'))
        bs = 2 * tp[idx].sum(1) / (npd[idx].sum(1) + nr[idx].sum(1))
        marg.append(dict(arm=arm, K=K, endpoint='lesion_f1', value=2 * tp.sum() / (npd.sum() + nr.sum()), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
    cd = per[per.arm == 'cdtr'].set_index(['K', 'patient_id'])
    for (arm, K), g in per.groupby(['arm', 'K']):
        if arm == 'cdtr' or K not in cd.index.get_level_values(0):
            continue
        g = g.set_index('patient_id').loc[pids]; c_ = cd.loc[K].loc[pids]
        for e in ENDPOINTS:
            dlt = c_[e].to_numpy() - g[e].to_numpy(); bs = dlt[idx].mean(1)
            paired.append(dict(comparison=f'cdtr - {arm}', K=K, endpoint=e, diff=dlt.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                               n_discordant=int((dlt != 0).sum())))
    return pd.DataFrame(marg), pd.DataFrame(paired)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='policy_validation'); a = ap.parse_args()
    c = cfg(); TAB.mkdir(exist_ok=True)
    df, truth = patient_rows(a.split, final=a.split == 'test')
    df.to_csv(TAB / f'{a.split}_patient_level_results.csv', index=False)
    summ = pd.DataFrame([dict(arm=arm, K=K, **summarise(g)) for (arm, K), g in df.groupby(['arm', 'K'])])
    summ.to_csv(TAB / f'{a.split}_summary.csv', index=False)
    marg, paired = boot(df, c['bootstrap']['n'], c['bootstrap']['seed'])
    sha = df.split_sha256.iloc[0]; csha = df.config_sha256.iloc[0]
    for t in (marg, paired):
        t['split'] = a.split; t['split_sha256'] = sha; t['config_sha256'] = csha
    marg.to_csv(TAB / f'{a.split}_metrics_with_95ci.csv', index=False)
    paired.to_csv(TAB / f'{a.split}_paired_differences.csv', index=False)
    if a.split == 'test':
        df.to_csv(TAB / 'patient_level_results.csv', index=False)
        marg.to_csv(TAB / 'metrics_with_95ci.csv', index=False)
    ref_counts = pd.Series([t['ref_response'] for t in truth.values()]).value_counts().to_dict()
    print('reference responses', ref_counts)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(summ.round(3).to_string(index=False))
