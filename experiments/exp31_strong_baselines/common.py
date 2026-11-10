"""Shared pieces of Exp 3.1 (pre-specified analysis plan). The MU-Glioma-Post multi-visit modules are used read-only."""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MV = REPO / 'experiments/multivisit_longitudinal'
sys.path.insert(0, str(MV))
import mu_glioma_adapter as A  # noqa: E402
import queries as Q  # noqa: E402
import timeline as T  # noqa: E402

OUT = REPO / 'outputs/exp31'
FACTOR = 1.6
SCALED = ('burden_ml', 'et_ml', 'netc_ml', 'snfh_ml')
REV_NOTE = ('The record of {v} has been revised: the measurement previously reported for {v} was wrong; the corrected record is now '
            'available and replaces it.')
_ORIG = A.visit_record


def cohort(min_visits=0):
    return {p: v for p, v in T.cohort().items() if len(v) >= min_visits}


def steps(visits, mode):
    if mode in ('natural', 'late'):
        return T.steps(visits, mode)
    if mode == 'revision':                                   # natural order; the revision of V2 arrives with V3
        return [(v, visits[:i + 1], f'rev:{visits[1]}' if i == 2 else '') for i, v in enumerate(visits)]
    raise ValueError(mode)


def effective_record(pid, tp, visits, current):
    """Revision mode: V2 is wrong (x FACTOR) until the step of V3, then corrected (with a revision note)."""
    rec, secs = _ORIG(pid, tp, visits)
    rec = copy.deepcopy(rec)
    if tp == visits[1]:
        if visits.index(current) < 2:
            for k in SCALED:
                if rec['measurements'].get(k) is not None:
                    rec['measurements'][k] = round(rec['measurements'][k] * FACTOR, 2)
        else:
            wrong = round(rec['measurements']['burden_ml'] * FACTOR, 2)
            rec['revision'] = dict(note='corrected record; replaces the measurement previously reported for this visit',
                                   previously_reported_burden_ml=wrong)
    return rec, secs


def install_revision():
    """Route every visit-record read (agent tools, OncoState source) through effective_record for the current timeline step."""
    def patched(pid, tp, visits):
        return effective_record(pid, tp, visits, A.VisitClock.get()['current'])
    A.visit_record = patched


def reference(pid, visits, avail, current, mode):
    recs = {v: (effective_record(pid, v, visits, current) if mode == 'revision' else _ORIG(pid, v, visits))[0] for v in avail}
    return Q.reference_answers([(v, recs[v]['measurements']['burden_ml']) for v in avail],
                               {v: recs[v]['treatments_reported_since_previous_visit'] for v in avail})
