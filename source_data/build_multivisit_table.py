#!/usr/bin/env python3
"""Source data for the MU-Glioma-Post multi-visit figure (long histories; 109 patients).

Concatenates the Experiment 3.1 result tables (outputs/exp31/results) in long format, without the event-sourced control system, with
values rounded to 4 decimals. Writes outputs/OncoState_Exp4_Exp5_full_without_event_sourced.csv.

    python source_data/build_multivisit_table.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / 'outputs/exp31/results'
OUT = REPO / 'outputs/OncoState_Exp4_Exp5_full_without_event_sourced.csv'

PARTS = [  # (experiment label, table)
    ('Exp4 strong baselines', 'A_aggregate.csv'),
    ('Exp4 strong baselines', 'A_paired.csv'),
    ('Exp4 strong baselines', 'A_consistency_by_depth.csv'),
    ('Exp4 strong baselines', 'A_late_arrival.csv'),
    ('Exp5 error propagation and recovery', 'B_aggregate.csv'),
]
COLUMNS = ['experiment', 'source_table', 'feature_set', 'landmark', 'n', 'auroc', 'brier', 'cal_slope', 'cal_intercept', 'auroc_lo',
           'auroc_hi', 'brier_lo', 'brier_hi', 'comparison', 'a', 'b', 'd_auroc', 'lo', 'hi', 'p_le0', 'analysis', 'scenario',
           'auroc_complete', 'auroc_degraded', 'drop', 'drop_lo', 'drop_hi', 'scope', 'condition', 'endpoint', 'changed_conclusions',
           'base_rate', 'cond_rate', 'diff', 'fixed', 'introduced', 'mcnemar_p', 'readers', 'accepted', 'reason', 'n_corrections', 'system',
           'metric', 'estimate', 'difference', 'visit_depth', 'mean', 'count', 'repeat_id', 'inserted_correctly', 'later_all_correct',
           'accuracy_after_arrival']


def main():
    frames = []
    for label, name in PARTS:
        d = pd.read_csv(RES / name)
        if 'system' in d:
            d = d[d.system != 'eventsourced']
        if 'comparison' in d:
            d = d[~d.comparison.str.contains('eventsourced')]
        d = d.round(4)
        d.insert(0, 'source_table', f'exp31/results/{name}')
        d.insert(0, 'experiment', label)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'wrote {OUT} ({len(out)} rows)')


if __name__ == '__main__':
    main()
