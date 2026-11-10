"""V0-equivalent correspondence energy (no learned parameters).

For a partial injective matching x between baseline lesions i and follow-up lesions j (registered centroids, distance d_ij):
    E(x) = sum_ij d_ij x_ij + (g/2) * (#unmatched baseline + #unmatched follow-up)
         = sum_ij (d_ij - g) x_ij + (g/2)(n + m)
which is exactly the objective the repository V0 MatchingNode minimises (Hungarian with dummy cost g/2 on both sides, g = 30 mm gate).
Pairs beyond the gate are NOT forbidden here (V0 forbids them); they have positive marginal cost d_ij - g, so the unconstrained argmin
is still V0, while long-distance correspondences (registration failures) remain in the hypothesis space at a finite energy.
We work with the relative energy F(x) = sum (d_ij - g) x_ij (the constant is dropped)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .rules import response, tenths

INF = 1e9


@dataclass
class Prepared:
    pid: str
    bids: list
    fids: list
    d: np.ndarray                 # n x m registered-centroid distances (mm)
    g: float
    bl_t: dict                    # diameters in tenths of mm (displayed 0.1 mm values)
    fu_t: dict
    v0_repository: list = field(default_factory=list)
    valid: bool = True
    invalid_reason: str = ''

    @property
    def n(self):
        return len(self.bids)

    @property
    def m(self):
        return len(self.fids)

    def F(self, pairs) -> float:
        bi, fi = {b: k for k, b in enumerate(self.bids)}, {f: k for k, f in enumerate(self.fids)}
        return float(sum(self.d[bi[a], fi[b]] - self.g for a, b in pairs))

    def response(self, pairs):
        return response(self.bl_t, self.fu_t, pairs)


def prepare(ev: dict, g: float = 30.0) -> Prepared:
    bl, fu = ev['baseline'], ev['followup']
    bids, fids = [l['lesion_id'] for l in bl], [l['lesion_id'] for l in fu]
    bc = np.array([l['centroid_mm'] for l in bl], float).reshape(-1, 3)
    fc = np.array([l['centroid_mm'] for l in fu], float).reshape(-1, 3)
    d = np.linalg.norm(bc[:, None, :] - fc[None, :, :], axis=2) if bids and fids else np.zeros((len(bids), len(fids)))
    P = Prepared(ev['patient_id'], bids, fids, d, g, {l['lesion_id']: tenths(l['diameter_mm']) for l in bl},
                 {l['lesion_id']: tenths(l['diameter_mm']) for l in fu}, [tuple(x) for x in ev.get('v0_pairs_repository', [])])
    why = []
    if len(set(bids)) != len(bids) or len(set(fids)) != len(fids):
        why.append('duplicate lesion id')
    if not np.all(np.isfinite(d)):
        why.append('non-finite centroid')
    if any(v <= 0 for v in list(P.bl_t.values()) + list(P.fu_t.values())):
        why.append('non-positive diameter')
    P.valid, P.invalid_reason = not why, '; '.join(why)
    return P


def best(P: Prepared, forced=(), no_partner=(), forbid=None):
    """Constrained argmin of F by an augmented assignment. forced: (b, f) pairs that must hold; no_partner: follow-up ids that must
    stay unmatched; forbid: optional ('pair', b, f) or ('unmatched', f) status to exclude (for min-marginals).
    Returns (pairs, F) or (None, inf) if infeasible."""
    n, m = P.n, P.m
    if n == 0 or m == 0:
        return [], 0.0
    N = n + m
    C = np.full((N, N), INF)
    C[:n, :m] = P.d - P.g
    C[np.arange(n), m + np.arange(n)] = 0.0          # baseline i unmatched
    C[n + np.arange(m), np.arange(m)] = 0.0          # follow-up j unmatched
    C[n:, m:] = 0.0
    bi, fi = {b: k for k, b in enumerate(P.bids)}, {f: k for k, f in enumerate(P.fids)}
    for b, f in forced:
        i, j = bi[b], fi[f]
        keep = C[i, j]
        C[i, :] = INF; C[:, j] = INF; C[i, j] = keep
    for f in no_partner:
        C[:n, fi[f]] = INF
    if forbid is not None:
        if forbid[0] == 'pair':
            C[bi[forbid[1]], fi[forbid[2]]] = INF
        else:
            C[n + fi[forbid[1]], fi[forbid[1]]] = INF
    r, c = linear_sum_assignment(C)
    if C[r, c].sum() >= INF / 2:
        return None, float('inf')
    pairs = sorted((P.bids[i], P.fids[j]) for i, j in zip(r, c) if i < n and j < m)
    return pairs, P.F(pairs)


def min_marginals(P: Prepared, pairs, forced=(), no_partner=()):
    """For each still-unrevealed follow-up lesion j: Delta_j = min F with j's current status excluded - F(pairs). Small Delta = the
    energy alone cannot decide j's identity (decision-agnostic uncertainty)."""
    base = P.F(pairs)
    fixed = {f for _, f in forced} | set(no_partner)
    partner = {f: b for b, f in pairs}
    out = {}
    for f in P.fids:
        if f in fixed:
            continue
        fb = ('pair', partner[f], f) if f in partner else ('unmatched', f)
        _, e = best(P, forced, no_partner, forbid=fb)
        out[f] = e - base
    return out
