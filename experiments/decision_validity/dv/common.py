"""Shared paths, loaders and the reference-isolation guard.

Only `Vault` reads the reference file. Decision code runs inside `no_reference()`, an audit hook that raises PermissionError when
anything opens a reference file, so a decision can never depend on the hidden correspondence of the patient being decided."""
from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
P1 = REPO / 'experiments/cdtr'
E26 = REPO / 'exp26_clinician_coevolution'
sys.path.insert(0, str(P1))
from cdtr import schema as S  # noqa: E402,F401
from cdtr.conformal import quantile  # noqa: E402
from cdtr.energy import Prepared, best, prepare  # noqa: E402,F401
from cdtr.verifier import verify  # noqa: E402,F401

CFG = yaml.safe_load((ROOT / 'configs/decision_validity.yaml').read_text())
REF_FILES = ('reference_HIDDEN.json', 'cohort_ground_truth', 'hybrid_oof')   # hybrid OOF is a decision input -> allowed below
PROTECTED = ('reference_HIDDEN.json', 'cohort_ground_truth')
_GUARD = {'on': False}


def _hook(event, args):
    if _GUARD['on'] and event == 'open' and args and isinstance(args[0], (str, Path)):
        s = str(args[0])
        if any(k in s for k in PROTECTED):
            raise PermissionError(f'reference access inside decision code: {s}')


sys.addaudithook(_hook)


@contextlib.contextmanager
def no_reference():
    prev = _GUARD['on']; _GUARD['on'] = True
    try:
        yield
    finally:
        _GUARD['on'] = prev


def evidence():
    return json.loads((P1 / 'results/evidence.json').read_text())


def roles():
    """Patient roles from the Exp 2.6 batch construction (the 274 = 50 + 80 + 144; 25 development)."""
    sys.path.insert(0, str(E26))
    from adapters.schema_mapping import load_bundle
    from evaluation.clinician_coevolution import SOURCE, make_batches
    from evaluation.cohorts import ALL_REVIEW_PATIENTS
    b = load_bundle(SOURCE)
    bt = make_batches(b['manifest'], json.loads(ALL_REVIEW_PATIENTS.read_text()))
    cohort = sorted(p for k in CFG['cohort'] for p in bt[k]); ext = sorted(p for k in CFG['external'] for p in bt[k])
    assert len(cohort) == 274 and len(ext) == 25 and not set(cohort) & set(ext)
    role = {p: k for k, ps in bt.items() for p in ps}
    return cohort, ext, role


def exp1_states():
    st = pd.read_csv(REPO / 'experiments/cdtr_exp1_state/results/per_patient_states.csv', dtype={'patient_id': str})
    return st[st.variant == 'crossfit']


def folds():
    st = exp1_states()
    return st.drop_duplicates('patient_id').set_index('patient_id').fold.astype(int).to_dict()


def stratum(ev_p: dict, cut: float | None = None, edges=None) -> str:
    """Reference-free evidence-quality stratum. single: one hypothesis only. Otherwise by registration residual."""
    if not ev_p['baseline'] or not ev_p['followup']:
        return 'single'
    r = ev_p.get('registration_residual_mm')
    if r is None or not np.isfinite(r):
        return 'single'
    if edges is not None:
        return f't{int(np.searchsorted(edges, r, side="right"))}'
    return 'reliable' if r <= (CFG['acceptance']['strata']['cut_primary_mm'] if cut is None else cut) else 'degraded'


def min_cal(alpha: float) -> int:
    return math.ceil((1 - alpha) / alpha - 1e-9)


class Vault:
    """The only reader of reference correspondences. Gives (i) nonconformity scores of calibration patients and (ii) final labels."""

    def __init__(self):
        if _GUARD['on']:
            raise PermissionError('Vault opened inside decision code')
        R = json.loads((P1 / 'results/reference_HIDDEN.json').read_text())
        self._pairs = {p: sorted(tuple(x) for x in v['pairs']) for p, v in R.items()}

    def pairs(self, pid):
        return list(self._pairs[pid])

    def scores(self, prep: dict, F0: dict, pids):
        return {p: prep[p].F(self._pairs[p]) - F0[p] for p in pids}

    def ref_response(self, P: Prepared):
        return P.response(self._pairs[P.pid])[0]


def q_tables(scores: dict, cohort, ext, fold, alphas, strata: dict | None = None):
    """Cross-fitted split-conformal q per patient: pool = other-fold cohort patients + external patients (same stratum if given).
    Returns {(alpha, pid): (q, pool_size)}; q = inf if the pool is too small for the rank (caller decides to abstain)."""
    out = {}
    for p in cohort:
        pool = [o for o in cohort if fold[o] != fold[p]] + list(ext)
        if strata is not None and strata[p] != 'single':
            pool = [o for o in pool if strata[o] == strata[p]]
        sc = [scores[o] for o in pool]
        for a in alphas:
            out[(a, p)] = (quantile(sc, a), len(sc))
    return out
