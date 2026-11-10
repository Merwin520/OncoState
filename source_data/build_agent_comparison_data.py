#!/usr/bin/env python3
"""Source data: OncoState vs vision-language models and tool-using agents on the CT cohort, hybrid matcher, efficiency and correction persistence.

    python build_agent_comparison_data.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import json
import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / 'experiments/exp1_agentic_longitudinal/outputs/exp22_external_final'
NAME = 'OncoState'
METRICS = [('match_f1', 'Match F1'), ('change_accuracy', 'Change\ndirection acc.'), ('burden_accuracy', 'Burden\ndirection acc.')]
SERVER = {NAME: 'Full OncoAgent', 'GPT-5.5': 'Direct VLM A (gpt-5.5)', 'MedGemma': 'Direct medical VLM (MedGemma 4B-it)',
          'LLaVA-Med': 'Direct medical VLM (LLaVA-Med v1.5 7B)'}
QWEN = {'Generic-memory agent': 'Generic stateful tool agent', 'Stateless agent': 'Stateless tool agent'}
ORDER = [NAME, 'GPT-5.5', 'MedGemma', 'LLaVA-Med']
RECOVERY = [('Direct VLM A (gpt-5.5)', 'Direct\nVLM'), ('Generic tool-using agent (gpt-5.2)', 'Generic\nagent'), ('Full OncoAgent', NAME),
            ('Conventional pipeline (registered centroid + Hungarian)', 'Conven-\ntional')]
ROOT = HERE.parent
CORR = [('OncoState', 'Durable state (OncoState)', ROOT / 'outputs/stateful_baseline/correction_persistence.csv'),
        ('Generic stateful tool agent', 'Generic memory agent', ROOT / 'outputs/stateful_baseline/correction_persistence.csv'),
        ('Stateless tool agent', 'Stateless agent', ROOT / 'outputs/stateful_baseline/correction_persistence.csv'),
        ('F_nondurable_correction', 'Non-durable state', ROOT / 'outputs/state_semantics_ablation/correction_retention_ct.csv')]


def build() -> pd.DataFrame:
    R = []

    def r(panel, series='', label='', x=np.nan, y=np.nan, lo=np.nan, hi=np.nan, value=np.nan, n=np.nan, note=''):
        R.append(dict(panel=panel, series=series, label=label, x=x, y=y, lo=lo, hi=hi, value=value, n=n, note=note))

    em = pd.read_csv(EXT / 'external_metrics.csv')
    pdiff = pd.read_csv(EXT / 'external_paired_differences.csv')
    est = lambda s, m: float(em[(em.method == SERVER[s]) & (em.metric == m)].iloc[0].estimate)  # noqa: E731
    for sysname in [s for s in ORDER if s in SERVER]:            # systems with case-level outputs only
        for m, _ in METRICS:
            q = em[(em.method == SERVER[sysname]) & (em.metric == m)].iloc[0]
            r('c', sysname, m, value=float(q.estimate), lo=q.ci95_lo, hi=q.ci95_hi, n=q.n_evaluated, note='exp22_external_final/external_metrics.csv')
            if sysname == NAME:
                continue
            q = pdiff[(pdiff.comparison == f'Full OncoAgent - {SERVER[sysname]}') & (pdiff.metric == m)].iloc[0]
            r('d', sysname, m, value=est(sysname, m) - est(NAME, m), lo=-q.ci95_hi, hi=-q.ci95_lo, note='paired patient-level bootstrap')
    SB = ROOT / 'outputs/stateful_baseline'
    qm = pd.read_csv(SB / 'external_metrics_same_definition.csv'); qp = pd.read_csv(SB / 'external_paired_same_definition.csv')
    for sysname, meth in QWEN.items():
        for m, _ in METRICS:
            q = qm[(qm.method == meth) & (qm.metric == m)].iloc[0]
            r('c', sysname, m, value=float(q.estimate), lo=float(q.ci95_lo), hi=float(q.ci95_hi), n=int(q.n_evaluated),
              note='outputs/stateful_baseline/external_metrics_same_definition.csv (Qwen3.5-9B, exp22x_eval)')
            q = qp[(qp.comparison == f'OncoState - {meth}') & (qp.metric == m)].iloc[0]
            r('d', sysname, m, value=-float(q.difference), lo=-float(q.ci95_hi), hi=-float(q.ci95_lo), note='paired patient-level bootstrap')
    hm = json.loads((ROOT / 'outputs/hybrid_matcher/run/results.json').read_text())   # hybrid matcher (5 x 5 CV)
    hyb = {'f1_all': ('Match F1, all', hm['baseline']['f1'], hm['hybrid']['f1']),
           'f1_reg': ('Match F1, residual > 15 mm', hm['baseline']['reg_sensitive_f1'], hm['hybrid']['reg_sensitive']['new']),
           'f1_new': ('New-lesion F1', hm['baseline']['class_f1']['newly_appearing'], hm['hybrid']['class_f1']['newly_appearing']),
           'f1_dis': ('Disappeared-lesion F1', hm['baseline']['class_f1']['disappearing'], hm['hybrid']['class_f1']['disappearing']),
           'fail': ('Complete matching failure (% patients)', 100 * hm['baseline']['complete_failure_rate'], 100 * hm['hybrid']['complete_failure_rate'])}
    for key, (lab, vb, vh) in hyb.items():
        r('hyb', 'frozen', key, value=vb, note=lab)
        r('hyb', 'hybrid', key, value=vh, note=lab)
    r('hyb', 'stat', 'n_reg_sensitive', value=hm['n_reg_sensitive'])
    fs = pd.read_csv(EXT / 'external_failure_summary.csv')
    fs = fs[fs.tag == 'registration_sensitive'].set_index('method')
    for method, lab in RECOVERY:
        r('e', 'recovery', lab, value=1 - float(fs.at[method, 'matching_error_rate']), n=int(fs.at[method, 'n_lesions']),
          note=f'1 - matching error rate on registration-sensitive lesions; {method}')
    ef = pd.read_csv(ROOT / 'outputs/stateful_baseline/efficiency_metrics.csv').set_index('method')   # same-model controlled run
    for key, lab in (('state_constructions_per_patient', 'State constructions'), ('total_tool_calls', 'Tool calls'),
                     ('llm_calls', 'LLM calls'), ('tokens', 'Tokens'), ('est_uncached_compute_sec', 'Uncached compute (s)')):
        for series, meth in (('oncostate', 'OncoState'), ('generic_memory', 'Generic stateful tool agent'), ('stateless', 'Stateless tool agent')):
            r('g', series, lab, value=float(ef.at[meth, key]), note='outputs/stateful_baseline/efficiency_metrics.csv')
    for key, lab, path in CORR:
        c = pd.read_csv(path, dtype={'patient_id': str})
        c = c[(c.method if 'method' in c else c.variant) == key]
        for q, g in c.groupby('query'):
            r('corr', lab, q, value=float(g.reflects_correction.mean()), n=int(len(g)), note=str(path.relative_to(ROOT)))
    return pd.DataFrame(R)


def main():
    out = Path(__file__).resolve().parent / 'out' / 'figure23_nature_data.csv'
    out.parent.mkdir(exist_ok=True)
    d = build()
    d.to_csv(out, index=False)
    print(f'wrote {out} ({len(d)} rows)')


if __name__ == '__main__':
    main()
