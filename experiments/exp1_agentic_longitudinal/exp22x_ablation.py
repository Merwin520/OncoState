"""Persistence ablation (kept strictly separate from the external system comparison): persistent state vs state-free execution with the SAME
underlying tools (segmentation, registration, matcher, measurement, status logic). Accuracy is NOT the endpoint here; the endpoints are
reconstruction/tool/LLM cost, cross-query consistency, provenance availability and edit retention. Reuses the exp22b outputs (agent v1 = same matcher)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

B1 = Path(__file__).resolve().parent / 'outputs/exp22_baselines'
B2 = Path(__file__).resolve().parent / 'outputs/exp22_external_final'


def main():
    eff = pd.read_csv(B1 / 'efficiency_comparison.csv'); st = pd.read_csv(B1 / 'baseline_statistics.csv'); cov = json.loads((B1 / 'agent_coverage.json').read_text())
    ret = pd.read_csv(B1 / 'edit_retention_summary.csv') if (B1 / 'edit_retention_summary.csv').exists() else pd.DataFrame()
    full = eff[eff.method == 'Full OncoAgent'].iloc[0]; ag = eff[eff.method == 'Generic state-free agent'].iloc[0]
    cons = lambda m: float(st[(st.quantity == 'cross_query_consistency') & (st.method == m)].value.iloc[0])
    rows = []
    def add(metric, persistent, ephemeral, note=''): rows.append(dict(metric=metric, persistent=persistent, ephemeral_state_free=ephemeral, note=note))
    add('state constructions per patient (4 queries)', float(full.state_constructions_per_patient), float(ag.state_constructions_per_patient), 'ephemeral = fresh episode per query')
    add('registration invocations', float(full.registration_invocations), float(ag.registration_invocations)); add('matching invocations', float(full.matching_invocations), float(ag.matching_invocations))
    add('measurement invocations', float(full.measurement_invocations), float(ag.measurement_invocations)); add('total tool calls', float(full.total_tool_calls), float(ag.total_tool_calls))
    add('LLM calls', float(full.llm_calls), float(ag.llm_calls)); add('tokens', float(full.tokens), float(ag.tokens)); add('LLM latency (s)', float(full.llm_latency_sec), float(ag.llm_latency_sec))
    add('estimated uncached compute (s)', float(full.est_uncached_compute_sec), float(ag.est_uncached_compute_sec), 'from mean measured per-tool times')
    add('cross-query consistency', cons('Full OncoAgent'), cons('Generic state-free agent')); add('provenance availability (grounded fraction)', 1.0, float(st[(st.quantity == 'provenance_grounded_fraction') & (st.method == 'Generic state-free agent')].value.iloc[0]))
    if len(ret):
        f = ret[ret.system == 'Full OncoAgent']; g = ret[ret.system == 'Generic state-free agent']
        add('edit retention in later queries (mean of Q1-Q3)', float(f['mean'].mean()) if len(f) else np.nan, float(g[g['query'] != 'Q1']['mean'].mean()) if len(g) else np.nan, 'ephemeral: Q2/Q3 are fresh episodes without the correction; Q1 (correction given in-episode) is reported in edit_retention.csv')
    out = pd.DataFrame(rows); out['agent_coverage'] = json.dumps(cov); out.to_csv(B2 / 'persistence_ablation_metrics.csv', index=False)
    for f in ('edit_retention.csv', 'edit_retention_summary.csv'):
        if (B1 / f).exists(): (B2 / f).write_bytes((B1 / f).read_bytes())
    print(out.drop(columns='agent_coverage').to_string(index=False))


if __name__ == '__main__':
    main()
