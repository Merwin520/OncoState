"""Shared simulated timeline: which visits are available at each step, and which queries are asked there (identical for every system).

Only the cohort SCHEDULE (visit ids in chronological order) is read here - never measurements or reference answers.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mu_glioma_adapter as A  # noqa: E402

OUT = HERE.parents[1] / 'outputs/multivisit_longitudinal'


def cohort():
    """{patient_id: [visits in chronological order]} for included patients (from cohort_manifest.csv)."""
    m = pd.read_csv(OUT / 'cohort_manifest.csv')
    m = m[m.included_or_excluded == 'included'].sort_values(['patient_id', 'visit_index'])
    return {p: list(g.visit_id) for p, g in m.groupby('patient_id')}


def steps(visits, mode='natural'):
    """[(current_visit, available_visits, tag)]; tag '' or 'late:<visit>'."""
    if mode == 'natural':
        return [(v, visits[:i + 1], '') for i, v in enumerate(visits)]
    if mode == 'late':                                  # hide V2; V1 -> V3 -> late V2 (at V3) -> V4 .. VT
        assert len(visits) >= 4
        v1, v2, v3 = visits[:3]
        out = [(v1, [v1], ''), (v3, [v1, v3], ''), (v3, [v1, v2, v3], f'late:{v2}')]
        out += [(v, visits[:i + 1], '') for i, v in enumerate(visits) if i >= 3]
        return out
    raise ValueError(mode)


def queries_for(pid, available, current):
    qs = ['Q1']
    if len(available) >= 2:
        qs += ['Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    if A.mri_day(pid, current) is not None:
        qs.append('Q7')
    if len(available) >= 2:
        qs.append('Q8')
    return qs


def qid(current, q, tag):
    return f'{current}|{q}' + (f'|{tag}' if tag else '')
