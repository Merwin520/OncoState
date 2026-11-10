"""Experiment 2.3 finalisation: main table, summary JSON, run manifest, and a printed failure-mode analysis."""
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

import exp23_llm_only as LL
import exp23_rules as RL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
SYSTEMS = ['Full OncoAgent', 'No-retrieval', 'LLM-only', 'Rule executor only']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    st = pd.read_csv(OUT / 'exp23_statistics.csv'); pred = pd.read_csv(OUT / 'benchmark_predictions_long.csv'); cases = pd.read_csv(OUT / 'rule_challenge_cases.csv')
    real = json.loads((OUT / 'real_summary.json').read_text()); dis = pd.read_csv(OUT / 'engine_vs_spec_disagreements.csv')
    ret = json.loads((OUT / 'guideline_retrieval.json').read_text())
    g = lambda m, s, sec='synthetic_main': st[(st.section == sec) & (st.metric == m) & (st.system == s)].iloc[0]
    fmt = lambda r: f'{r.estimate:.3f}' + (f' [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]' if not np.isnan(r.ci95_lo) else '')
    rows = []
    for s in SYSTEMS:
        rows.append({'system': s, 'n_cases': int(g('accuracy', s).n), 'response_accuracy': fmt(g('accuracy', s)), 'accuracy_complete_evidence': fmt(g('acc_complete', s)),
                     'macro_precision': f"{g('macro_precision', s).estimate:.3f}", 'macro_recall': f"{g('macro_recall', s).estimate:.3f}", 'weighted_f1': f"{g('weighted_f1', s).estimate:.3f}", 'macro_f1': f"{g('macro_f1', s).estimate:.3f}",
                     'cohen_kappa': f"{g('kappa', s).estimate:.3f}", 'rule_correctness': fmt(g('rule', s)), 'traceability_complete': fmt(g('trace', s)), 'citation_supported': fmt(g('cit_supported', s)),
                     'abstention_correct': f"{g('abstention_correct', s).estimate:.3f}", 'unsupported_decision_rate': f"{g('unsupported', s).estimate:.3f}", 'false_abstention_rate': f"{g('false_abstain', s).estimate:.3f}"})
    pd.DataFrame(rows).to_csv(OUT / 'exp23_main_table.csv', index=False)
    # failure modes
    main_ = pred[pred.subset == 'main']; fm = {}
    for s in SYSTEMS:
        d = main_[main_.system == s]; e = d[~d.correct].groupby('challenge_type').size().sort_values(ascending=False)
        fm[s] = {'errors_by_type': e.head(6).to_dict(), 'n_errors_repeat_level': int((~d.correct).sum()), 'predicted_when_wrong': d[~d.correct].predicted_response.value_counts().to_dict()}
    stress = pred[pred.subset == 'threshold_stress']; fm['stress_errors'] = {s: int((~stress[stress.system == s].correct).sum()) for s in SYSTEMS}
    summary = dict(
        real_hcuch=real, synthetic=dict(main_instances=int((cases.subset == 'main').sum()), threshold_stress_instances=int((cases.subset == 'threshold_stress').sum()), total=len(cases),
                                        expected_response_counts=cases.expected_response.value_counts().to_dict(), challenge_types=cases.groupby('challenge_type').size().to_dict()),
        engine_vs_spec=dict(n_disagreements=len(dis), all_at_exact_boundary=bool(len(dis) and (dis.challenge_type.isin(['E_near_PR_threshold', 'S1_PR_boundary']).all())),
                            by_type=dis.challenge_type.value_counts().to_dict() if len(dis) else {}, note='float comparison 0.70*baseline in the repository engine; spec uses exact rational arithmetic'),
        retrieval=dict(model=ret['model'], retrieved_utc=ret['retrieved_utc'], clauses={k: dict(citation_correct=v['citation_correct'], citation_supported=v['citation_supported'], sources=[c['url'] for c in v['citations']]) for k, v in ret['clauses'].items()}),
        main_table=rows, failure_modes=fm)
    (OUT / 'exp23_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    model_ver = None
    try:
        from openai import OpenAI
        model_ver = OpenAI().responses.create(model=LL.MODEL, input='ok', max_output_tokens=20).model
    except Exception as e:
        model_ver = f'unavailable ({type(e).__name__})'
    files = sorted(ROOT.glob('exp23_*.py'))
    manifest = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), dataset=dict(hcuch_root=os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'HCUCH RECIST CT/final-formatted/final-formatted'), files=['recist_measurements.csv', 'metadata/patients.csv', 'metadata/series.json', 'images/*/*/masks', 'images/*/*/labels']),
                    repository=str(ROOT.parents[1]), git_commit='not a git repository (code identified by SHA-256 below)', code_sha256={f.name: sha(f) for f in files},
                    llm=dict(requested_model=LL.MODEL, api_reported_model=model_ver, repeats=LL.REPEATS, prompt_sha256_16=LL.PROMPT_HASH, api='OpenAI Responses API', decoding='API defaults (no temperature override)'),
                    guideline=dict(name='RECIST 1.1', reference='Eisenhauer EA et al. Eur J Cancer 2009;45:228-247', retrieval_model=ret['model'], retrieval_tool='web_search', retrieved_utc=ret['retrieved_utc'],
                                   sources=sorted({c['url'] for v in ret['clauses'].values() for c in v['citations']})),
                    rule_engine=dict(path=str(RL._ENGINE_PATH), sha256_16=sha(RL._ENGINE_PATH)), random_seeds=dict(case_generation=20260921, bootstrap=20260921, case_shuffle_llm_pilot=1),
                    sample_counts=dict(hcuch_measurement_rows=real['n_measurement_rows'], hcuch_patients_with_measurements=real['n_patients_with_measurements'], hcuch_studies=real['n_studies_with_measurements'],
                                       real_evaluable_cases=real['n_real_cases'], real_evaluable_patients=real['n_patients_evaluable'], synthetic_main=summary['synthetic']['main_instances'],
                                       threshold_stress=summary['synthetic']['threshold_stress_instances'], llm_calls=int(len(pred[pred.system == 'LLM-only']))),
                    not_done=['iRECIST (engine and data do not support it)', 'overall RECIST response on real HCUCH (no new-lesion / non-target evidence)', 'expert adjudication of citations (mechanical checks only)'])
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=1))
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).T.to_string()); print(json.dumps(fm, indent=1)); print('disagreements', len(dis), summary['engine_vs_spec']['by_type'])


if __name__ == '__main__':
    main()
