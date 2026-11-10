"""Training-free Gumbel-Sinkhorn layer over the V0 energy (Mena et al. 2018), on the same augmented assignment matrix as energy.best:
real pair cost d_ij - g, 'unmatched' dummies cost 0, forbidden cells +inf. Log-potentials = -cost / tau.
  sinkhorn_marginals : entropic (Sinkhorn) doubly-stochastic matrix -> soft lesion-status probabilities
  gumbel_samples     : hard Gumbel-matching samples (tau -> 0 limit of Gumbel-Sinkhorn; perturb-and-MAP with i.i.d. Gumbel noise)
The only parameter is the temperature tau (mm), fitted on non-evaluation patients. Sampling explores the hypothesis space but never
proves anything: equivalence proofs come from the MILP verifier only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from .energy import INF, Prepared


def augmented_cost(P: Prepared):
    n, m = P.n, P.m
    N = n + m
    C = np.full((N, N), np.inf)
    C[:n, :m] = P.d - P.g
    C[np.arange(n), m + np.arange(n)] = 0.0
    C[n + np.arange(m), np.arange(m)] = 0.0
    C[n:, m:] = 0.0
    return C


def sinkhorn_marginals(P: Prepared, tau: float, iters=2000, tol=1e-7):
    """Returns (Pb, ub, uf): Pb[i, j] = P(baseline i <-> follow-up j); ub[i] = P(i unmatched); uf[j] = P(j unmatched)."""
    n, m = P.n, P.m
    if n == 0 or m == 0:
        return np.zeros((n, m)), np.ones(n), np.ones(m)
    L = -augmented_cost(P) / tau
    u = np.zeros(L.shape[0]); v = np.zeros(L.shape[1])
    for _ in range(iters):
        u = -logsumexp(L + v[None, :], axis=1)
        v_new = -logsumexp(L + u[:, None], axis=0)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new; break
        v = v_new
    X = np.exp(L + u[:, None] + v[None, :])
    return X[:n, :m], X[np.arange(n), m + np.arange(n)], X[n + np.arange(m), np.arange(m)]


def gumbel_samples(P: Prepared, tau: float, S: int, rng):
    n, m = P.n, P.m
    if n == 0 or m == 0:
        return [[] for _ in range(S)]
    C = augmented_cost(P)
    fin = np.isfinite(C)
    base = np.where(fin, C, INF)
    out = []
    for _ in range(S):
        G = rng.gumbel(size=C.shape)
        r, c = linear_sum_assignment(np.where(fin, base - tau * G, INF))
        out.append(sorted((P.bids[i], P.fids[j]) for i, j in zip(r, c) if i < n and j < m))
    return out


def status_nll(P: Prepared, tau: float, ref_pairs, eps=1e-9):
    """Mean negative log-likelihood of every detected lesion's reference status (partner or unmatched) under Sinkhorn marginals."""
    Pb, ub, uf = sinkhorn_marginals(P, tau)
    bi, fi = {b: k for k, b in enumerate(P.bids)}, {f: k for k, f in enumerate(P.fids)}
    pb = {a: b for a, b in ref_pairs}; pf = {b: a for a, b in ref_pairs}
    ll = [np.log(max(Pb[bi[a], fi[pb[a]]] if a in pb else ub[bi[a]], eps)) for a in P.bids]
    ll += [np.log(max(Pb[bi[pf[b]], fi[b]] if b in pf else uf[fi[b]], eps)) for b in P.fids]
    return -float(np.sum(ll)), len(ll)
