"""Experiment 4 shared layer: reuses the validated longitudinal-forecasting data layer (official Exp 2.8 synchronize/revise, I-SPY2/I-SPY1 evidence
sources, shared feature function F) unchanged."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E4 = HERE.parent
REPO = E4.parents[1]
E29 = REPO / 'experiments/longitudinal_forecasting'
sys.path.insert(0, str(E29))
from data import ISPY1, ISPY2, M2, T1, T2, F, table_from_memory, _f  # noqa: E402,F401

RES, TAB, FIG = E4 / 'results', E4 / 'tables', E4 / 'figures'
SEED, NB = 20261011, 2000
