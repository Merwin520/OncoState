"""CDT-R for longitudinal glioma MRI: candidate measurement / temporal states, the RANO-derived volumetric rule, and the exact
minimum-energy search for a state with a given category (analysis plan, Implementation details).

A decision = the current follow-up scan t, its priors (index 0 = post-operative baseline, then earlier follow-ups) and the 12-week flag.
State X = log-volumes u = log(V + 0.1 mL) of CE and T2 at t and at every prior, plus the exception state z (12-week window only).
Energy F(X) = sum |u - m| / sigma (+ lambda if z = 1). The point estimate (u = m, z = 0) has F = 0.
min_cost(d, k) = min F(X) subject to g(X) = k, computed exactly in the current-scan value (distance to an interval) and on a u-grid in
the nadir level L and the baseline value bv; the grid minimum over-estimates the true minimum by at most d.delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 0.1
LOG_CR = np.log(0.1 + EPS)
U_MIN = np.log(EPS)
STEP = 0.02
CATS = ('PD', 'SD', 'PR', 'CR')


def to_u(v):
    return np.log(np.asarray(v, float) + EPS)


def to_v(u):
    return np.exp(np.asarray(u, float)) - EPS


def H(L):
    """PD threshold of the current value for a nadir at u-level L: V >= max(1.4 Vn, Vn + 1 mL)."""
    vn = np.maximum(to_v(L), 0.0)
    return np.log(np.maximum(1.4 * vn, vn + 1.0) + EPS)


def P(bv):
    """PR threshold for a baseline at u-level bv: V <= 0.35 Vb."""
    return np.log(0.35 * np.maximum(to_v(bv), 0.0) + EPS)


def rule(v_t: dict, priors: list[dict], lt3m: bool, z: int = 0) -> str:
    """RANO-derived volumetric rule on volumes (mL). priors[0] = baseline."""
    pd_ = False
    for c in ('ce', 't2'):
        n = min(p[c] for p in priors)
        pd_ |= v_t[c] >= max(1.4 * n, n + 1.0)
    if pd_ and (not lt3m or z == 1):
        return 'PD'
    if v_t['ce'] < 0.1:
        return 'CR'
    if v_t['ce'] <= 0.35 * priors[0]['ce']:
        return 'PR'
    return 'SD'


@dataclass
class Decision:
    key: tuple
    m_t: dict                      # u of the current scan per compartment
    s_t: dict                      # sigma of the current scan
    m_p: dict                      # compartment -> array of prior u (index 0 = baseline)
    s_p: dict
    lt3m: bool
    lam: float = 1.0
    fixed_priors: bool = False     # comparator C3: only the current scan may move
    allow_z: bool = True
    cost: dict = field(default_factory=dict)
    delta: float = 0.0

    def point(self):
        return rule({c: float(to_v(self.m_t[c])) for c in ('ce', 't2')},
                    [{c: float(to_v(self.m_p[c][i])) for c in ('ce', 't2')} for i in range(len(self.m_p['ce']))], self.lt3m)


def _dist(m, lo, hi):
    """distance from m to [lo, hi] (inf if empty), elementwise."""
    d = np.where(m < lo, lo - m, np.where(m > hi, m - hi, 0.0))
    return np.where(lo <= hi, d, np.inf)


def _grid(m_all):
    """uniform u-grid (spacing <= STEP) that also contains every measured value exactly, so the point estimate has cost 0."""
    top = max(np.max(m_all) + 3.0, LOG_CR + 3.0)
    return np.unique(np.concatenate([np.arange(U_MIN, top + STEP, STEP), np.asarray(m_all, float)]))


def _nadir_free(m, s, G):
    """N(L): min cost of the priors for nadir exactly L (all priors symmetric). shape (len(G),)"""
    up = np.maximum(G[None, :] - m[:, None], 0.0) / s[:, None]
    base = up.sum(0)
    need = (m[:, None] > G[None, :]).all(0)
    lower = ((m[:, None] - G[None, :]) / s[:, None]).min(0)
    return base + np.where(need, lower, 0.0)


def _nadir_ce(m, s, G):
    """N(L, bv): baseline (index 0) at bv, other priors at least L, nadir exactly L. shape (len(G), len(G))"""
    mb, sb, mo, so = m[0], s[0], m[1:], s[1:]
    cb = np.abs(G - mb) / sb                                              # over bv
    if len(mo):
        base = (np.maximum(G[None, :] - mo[:, None], 0.0) / so[:, None]).sum(0)          # over L
        need = (mo[:, None] > G[None, :]).all(0)
        lower = ((mo[:, None] - G[None, :]) / so[:, None]).min(0)
        extra = np.where(need, lower, 0.0)
    else:
        base = np.zeros_like(G); extra = np.full_like(G, np.inf)
    Lg, Bg = np.meshgrid(np.arange(len(G)), np.arange(len(G)), indexing='ij')
    N = base[:, None] + cb[None, :] + np.where(Bg > Lg, extra[:, None], 0.0)
    return np.where(Bg >= Lg, N, np.inf)


def solve(d: Decision) -> Decision:
    """Fill d.cost[k] = min F subject to g = k for every category, and d.delta."""
    if d.fixed_priors:                                                    # C3: priors and z at the point estimate
        mp = {c: d.m_p[c] for c in d.m_p}
        out = {}
        nad = {c: float(np.min(mp[c])) for c in ('ce', 't2')}
        hpd = {c: float(H(nad[c])) for c in ('ce', 't2')}
        lt = d.lt3m
        mt, st = d.m_t, d.s_t
        pd_cost = min(_dist(mt[c], hpd[c], np.inf) / st[c] for c in ('ce', 't2'))
        out['PD'] = np.inf if lt else float(pd_cost)
        cap_ce = np.inf if lt else hpd['ce']
        np_t2 = 0.0 if lt else float(_dist(mt['t2'], U_MIN, hpd['t2']) / st['t2'])
        pb = float(P(mp['ce'][0]))
        out['CR'] = float(_dist(mt['ce'], U_MIN, min(LOG_CR, cap_ce)) / st['ce']) + np_t2
        out['PR'] = float(_dist(mt['ce'], LOG_CR, min(pb, cap_ce)) / st['ce']) + np_t2
        out['SD'] = float(_dist(mt['ce'], max(LOG_CR, pb), cap_ce) / st['ce']) + np_t2
        d.cost, d.delta = out, 0.0
        return d
    G = _grid(np.concatenate([d.m_p['ce'], d.m_p['t2'], [d.m_t['ce'], d.m_t['t2']]]))
    HG, PG = H(G), P(G)
    # T2: nadir over all priors
    N2 = _nadir_free(d.m_p['t2'], d.s_p['t2'], G)
    pd_t2 = float(np.min(N2 + _dist(d.m_t['t2'], HG, np.inf) / d.s_t['t2']))
    np_t2 = float(np.min(N2 + _dist(d.m_t['t2'], U_MIN, HG) / d.s_t['t2']))
    # CE: nadir L and baseline bv
    N1 = _nadir_ce(d.m_p['ce'], d.s_p['ce'], G)
    pd_ce = float(np.min(N1.min(1) + _dist(d.m_t['ce'], HG, np.inf) / d.s_t['ce']))
    mt = d.m_t['ce']; st = d.s_t['ce']

    def ce_cat(cap):                                                      # cap[L] = upper bound of a from non-PD (or inf)
        capL = cap[:, None]
        cr = np.min(N1 + _dist(mt, U_MIN, np.minimum(LOG_CR, capL)) / st)
        pr = np.min(N1 + _dist(mt, LOG_CR, np.minimum(PG[None, :], capL)) / st)
        sd = np.min(N1 + _dist(mt, np.maximum(LOG_CR, PG[None, :]), capL) / st)
        return {'CR': float(cr), 'PR': float(pr), 'SD': float(sd)}

    if d.lt3m:
        free = ce_cat(np.full_like(G, np.inf))
        out = {k: v for k, v in free.items()}
        out['PD'] = (d.lam + min(pd_ce, pd_t2)) if d.allow_z else np.inf
    else:
        nonpd = ce_cat(HG)
        out = {k: v + np_t2 for k, v in nonpd.items()}
        out['PD'] = min(pd_ce, pd_t2)
    inv = lambda x: float(np.sum(1.0 / x))  # noqa: E731
    d.delta = STEP / 2 * (inv(d.s_p['ce']) + 1.5 / d.s_t['ce'] + 1.0 / d.s_p['ce'][0] + 1.0 / d.s_t['ce']
                          + inv(d.s_p['t2']) + 1.5 / d.s_t['t2'])
    d.cost = out
    return d


def flip_cost(d: Decision) -> tuple[float, str]:
    """min energy of a state whose category differs from the point category, and that category."""
    k0 = d.point()
    alt = {k: v for k, v in d.cost.items() if k != k0}
    k = min(alt, key=alt.get)
    return alt[k], k
