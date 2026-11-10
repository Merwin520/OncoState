"""Experiment 2.2 external comparison: unified evaluation at the level of REFERENCE lesions.

Every system's answer is converted to reference-lesion space:
  * Full / conventional / oracle: predicted lesion ids -> reference lesions through the frozen IoU>0.1 mapping of Experiment 2.2;
  * direct multimodal models and the generic agent: packet ids B#/F# -> reference lesions through the private packet key.
A predicted pair (gb, gf) is correct iff gb == gf (the frozen 2.2 identity reference: a lesion id shared by baseline and follow-up masks).
Change labels are increased / stable / decreased (>+20% / within +-20% / <-20% volume, the frozen 2.2 rule); burden = direction of the change
in total lesion volume. Patient-level bootstrap is used everywhere (never lesion-level).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import exp22b_common as C

BASE = C.BASE.parent / 'exp22_external_final'
CLS = ['increased', 'stable', 'decreased']
CH = {'enlarged': 'increased', 'stable': 'stable', 'shrunk': 'decreased'}
BU = {'increase': 'increased', 'no_change': 'stable', 'decrease': 'decreased'}
NB, SEED = 5000, 20260921
METRICS = ['match_precision', 'match_recall', 'match_f1', 'change_accuracy', 'change_accuracy_conditional', 'change_weighted_f1', 'change_macro_f1', 'change_kappa',
           'burden_accuracy', 'burden_balanced_accuracy', 'burden_weighted_f1', 'burden_kappa', 'change_uncertain_rate', 'burden_uncertain_rate']
D = 7 + 15 + 12 + 12


def det_pred(r):
    """Deterministic method result (from exp22b_methods) -> reference-lesion prediction."""
    ref, pred = r['ref'], r['pred']; pairs, change = [], {}
    for a, b in pred['pairs']:
        gb, gf = ref['mb'].get(a), ref['mf'].get(b); pairs.append((gb, gf))
        if pred['change'].get((a, b)): change[(gb, gf)] = CH[pred['change'][(a, b)]]
    return dict(pairs=pairs, change=change, burden=BU.get(pred['burden']), valid=True)


def packet_pred(ans, key):
    """Structured answer with packet ids -> reference-lesion prediction; ans None = invalid output (empty prediction, counted wrong)."""
    if ans is None: return dict(pairs=[], change={}, burden=None, valid=False)
    pairs, change = [], {}
    for m in ans['matches']:
        if m['followup_id'] == 'no_match': continue
        gb, gf = key['baseline'].get(m['baseline_id']), key['followup'].get(m['followup_id']); pairs.append((gb, gf))
        if m['change'] in CLS + ['uncertain']: change[(gb, gf)] = m['change']
    return dict(pairs=pairs, change=change, burden=ans['burden_direction'], valid=True)


def evaluate(ref, pred):
    """Per-patient sufficient statistics (counts), as one vector of length D."""
    gt_pers = [g for g, c in ref['gclass'].items() if c == 'persistent' and g in ref['gchg']]
    pairs = pred['pairs']; tp = sum(1 for gb, gf in pairs if gb is not None and gb == gf); correct = {gb for gb, gf in pairs if gb is not None and gb == gf}
    unc = np.zeros((3, 5)); cond = np.zeros((3, 4)); unc_ok = cond_ok = cond_n = 0
    for g in gt_pers:
        gtc = CLS.index(CH[ref['gchg'][g]])
        if g in correct:
            pc = pred['change'].get((g, g)); col = CLS.index(pc) if pc in CLS else 3
            unc[gtc, col] += 1; cond[gtc, col] += 1; cond_n += 1; ok = (col == gtc); cond_ok += ok; unc_ok += ok
        else: unc[gtc, 4] += 1
    bur = np.zeros((3, 4)); bg = CLS.index(BU[ref['gburden']]); bp = CLS.index(pred['burden']) if pred.get('burden') in CLS else 3; bur[bg, bp] += 1
    return np.concatenate([[tp, len(pairs), len([g for g, c in ref['gclass'].items() if c == 'persistent']), unc_ok, len(gt_pers), cond_ok, cond_n], unc.ravel(), cond.ravel(), bur.ravel()]).astype(float)


def _f1_kappa(conf):
    """conf (..., 3, K>=3): rows reference classes, columns predictions (extra columns = uncertain / invalid, counted wrong)."""
    K = conf.shape[-1]; tp = np.stack([conf[..., i, i] for i in range(3)], -1); w = conf.sum(-1); fn = w - tp; fp = conf[..., :3].sum(-2) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * tp / (2 * tp + fp + fn); wf1 = np.nansum(f1 * w, -1) / w.sum(-1); mf1 = np.nanmean(f1, -1)
        pad = np.zeros(conf.shape[:-2] + (K, K)); pad[..., :3, :] = conf; n = pad.sum((-2, -1)); po = np.trace(pad, axis1=-2, axis2=-1) / n; pe = (pad.sum(-1) * pad.sum(-2)).sum(-1) / n ** 2
        acc = np.trace(conf[..., :3], axis1=-2, axis2=-1) / conf.sum((-2, -1)); rec = np.stack([conf[..., i, i] / w[..., i] for i in range(3)], -1)
    return acc, wf1, mf1, (po - pe) / (1 - pe), np.nanmean(rec, -1)


def metrics(S):
    """S: (..., D) summed sufficient statistics -> dict of arrays."""
    o = {}; tp, pp, gp = S[..., 0], S[..., 1], S[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        o['match_precision'] = tp / pp; o['match_recall'] = tp / gp; o['match_f1'] = 2 * tp / (pp + gp)
        o['change_accuracy'] = S[..., 3] / S[..., 4]; o['change_accuracy_conditional'] = S[..., 5] / S[..., 6]
        cond = S[..., 22:34].reshape(S.shape[:-1] + (3, 4)); _, wf1, mf1, kap, _ = _f1_kappa(cond); o['change_weighted_f1'], o['change_macro_f1'], o['change_kappa'] = wf1, mf1, kap
        o['change_uncertain_rate'] = cond[..., 3].sum(-1) / cond.sum((-2, -1))
        b = S[..., 34:46].reshape(S.shape[:-1] + (3, 4)); acc, wf1, _, kap, bal = _f1_kappa(b)
        o['burden_accuracy'], o['burden_weighted_f1'], o['burden_kappa'], o['burden_balanced_accuracy'] = acc, wf1, kap, bal
        o['burden_uncertain_rate'] = b[..., 3].sum(-1) / b.sum((-2, -1))
    return o


def bootstrap(mats, avail, patients, nb=NB, seed=SEED):
    """mats {method: (n, D)}, avail {method: bool (n,)}; shared patient resamples; returns point, boot and the resample counts."""
    rng = np.random.default_rng(seed); n = len(patients); idx = rng.integers(0, n, size=(nb, n))
    counts = np.zeros((nb, n)); np.add.at(counts, (np.repeat(np.arange(nb), n), idx.ravel()), 1)
    point, boot = {}, {}
    for m in mats:
        X = mats[m] * avail[m][:, None]; point[m] = {k: float(v) for k, v in metrics(X.sum(0)).items()}; boot[m] = {k: np.asarray(v) for k, v in metrics(counts @ X).items()}
    return point, boot, counts


def paired_common(mats, avail, a, b, counts):
    """Metrics of a and b on the patients evaluated for BOTH (common subset), on the shared resamples."""
    com = (avail[a] & avail[b]).astype(float)[:, None]; Xa, Xb = mats[a] * com, mats[b] * com
    pa, pb = metrics(Xa.sum(0)), metrics(Xb.sum(0)); ba, bb = metrics(counts @ Xa), metrics(counts @ Xb)
    rows = []
    for k in METRICS:
        d = np.asarray(ba[k]) - np.asarray(bb[k]); rows.append(dict(comparison=f'{a} - {b}', metric=k, difference=float(pa[k] - pb[k]), ci95_lo=float(np.nanpercentile(d, 2.5)), ci95_hi=float(np.nanpercentile(d, 97.5)), n_common_patients=int(com.sum())))
    return rows
