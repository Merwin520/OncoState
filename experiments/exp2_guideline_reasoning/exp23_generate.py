"""Experiment 2.3-B/C case generator: synthetic rule-challenge instances (NOT patients).

Numeric templates (baseline sums, target-lesion counts) are sampled from the real HCUCH baseline studies; rule-relevant variables are then
varied systematically. Every instance has a reference response from the independent RECIST 1.1 specification (exp23_rules.spec_response)
and, where the generator states one, a design-intended response; disagreements are reported.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

import exp23_rules as RL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
SEED = 20260921
DATA = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'HCUCH RECIST CT/final-formatted/final-formatted'))
rng = np.random.default_rng(SEED)

m = pd.read_csv(DATA / 'recist_measurements.csv')
base = m[m.study_order == 'baseline'].groupby('patient_id').agg(s=('recist_measurement_mm', 'sum'), n=('recist_measurement_mm', 'count'))
TEMPL = [(int(pid), float(r.s), int(r.n)) for pid, r in base.iterrows()]
NT_OK = ['NON-CR/NON-PD', 'NON-CR/NON-PD', 'ABSENT', 'CR']


def r1(x):
    return round(float(x), 1)


def template():
    pid, s, n = TEMPL[rng.integers(len(TEMPL))]
    B = max(20, int(round(s * rng.uniform(0.8, 1.25))))
    return pid, B, n


def mk(ctype, family, history, cur, new=False, nt='NON-CR/NON-PD', design=None, tpl=None, n_l=1, **kw):
    c = dict(challenge_type=ctype, family=family, history=history, current_sum=cur, new_lesions=new, non_target=nt, design_expected=design,
             template_source=f'HCUCH_patient_{tpl[0]}' if tpl else '', n_target_lesions=n_l, all_disappeared=False, nodes_lt10=False, lesions_missing=False)
    c.update(kw)
    return c


def build_main():
    cases = []
    def add(k, fn):
        for _ in range(k):
            for _try in range(200):
                c = fn()
                if c is not None:
                    cases.append(c); break
    def tgt(c):
        return RL.spec_target(c['history'], c['current_sum'])[0]

    def A():
        t = template(); B = t[1]; cur = r1(B * (1 + rng.uniform(-.70, -.38)))
        return mk('A_clear_PR', 'main', [B], cur, False, str(rng.choice(['NON-CR/NON-PD', 'ABSENT'])), 'PR', t, t[2])
    def Bf():
        t = template(); B = t[1]; c = mk('B_clear_SD', 'main', [B], r1(B * rng.uniform(.82, 1.12)), False, str(rng.choice(NT_OK[:3])), 'SD', t, t[2]); return c if tgt(c) == 'SD' else None
    def C():
        t = template(); B = t[1]; return mk('C_clear_PD_size', 'main', [B], r1(B * rng.uniform(1.3, 2.4)), False, str(rng.choice(NT_OK[:3])), 'PD', t, t[2])
    def D():
        t = template(); B = t[1]; return mk('D_new_lesion_override', 'main', [B], r1(B * rng.uniform(.35, .75)), True, str(rng.choice(NT_OK[:3])), 'PD', t, t[2])
    def E():
        t = template(); B = t[1]; return mk('E_near_PR_threshold', 'main', [B], r1(B * (0.70 + rng.uniform(-.015, .015))), False, 'NON-CR/NON-PD', None, t, t[2])
    def F():
        t = template(); N = max(30, t[1]); return mk('F_near_PD_threshold_pct', 'main', [N], r1(N * (1.2 + rng.uniform(-.02, .02))), False, 'NON-CR/NON-PD', None, t, t[2])
    def G():
        N = int(rng.integers(10, 24)); cur = r1(rng.uniform(1.2 * N, N + 4.9)); t = template()
        return mk('G_pct_increase_insufficient_abs', 'main', [N], cur, False, 'NON-CR/NON-PD', 'SD', t, 1) if cur >= 1.2 * N and cur - N < 5 else None
    def H():
        t = template(); B = t[1]; nad = r1(B * rng.uniform(.45, .75))
        if rng.random() < .6:
            cur = r1(nad * rng.uniform(1.22, 1.5)); return mk('H_nadir_dependent', 'main', [B, nad], cur, False, 'NON-CR/NON-PD', 'PD' if cur - nad >= 5 else None, t, t[2])
        cur = r1(nad * rng.uniform(1.0, 1.15)); c = mk('H_nadir_dependent', 'main', [B, nad], cur, False, 'NON-CR/NON-PD', None, t, t[2]); return c
    def I():
        t = template(); B = t[1]; nt = str(rng.choice(['CR', 'ABSENT', 'NON-CR/NON-PD']))
        return mk('I_complete_response_pattern', 'main', [B], 0.0, False, nt, 'CR' if nt != 'NON-CR/NON-PD' else 'PR', t, t[2], all_disappeared=True, nodes_lt10=True)
    def J():
        t = template(); B = t[1]; v = rng.integers(5)
        if v == 0: return mk('J_conflicting_target_vs_nontarget', 'main', [B], r1(B * rng.uniform(.4, .65)), False, 'PD', 'PD', t, t[2])
        if v == 1: return mk('J_conflicting_target_vs_nontarget', 'main', [B], r1(B * rng.uniform(.85, 1.1)), False, 'PD', 'PD', t, t[2])
        if v == 2: return mk('J_conflicting_target_vs_nontarget', 'main', [B], r1(B * rng.uniform(.4, .65)), False, 'NON-CR/NON-PD', 'PR', t, t[2])
        if v == 3: return mk('J_conflicting_target_vs_nontarget', 'main', [B], r1(B * rng.uniform(1.3, 2.0)), False, 'CR', 'PD', t, t[2])
        return mk('J_conflicting_target_vs_nontarget', 'main', [B], r1(B * rng.uniform(.4, .65)), False, 'CR', 'PR', t, t[2])
    def K():
        t = template(); B = t[1]; v = rng.random()
        cur = r1(B * (rng.uniform(.4, .65) if v < .4 else rng.uniform(.85, 1.1) if v < .8 else rng.uniform(1.3, 2.0)))
        c = mk('K_missing_new_lesion_info', 'main', [B], cur, None, 'NON-CR/NON-PD', 'PD' if v >= .8 else 'NE', t, t[2]); return c
    def L():
        t = template(); B = t[1]; return mk('L_missing_baseline', 'main', None, r1(B * rng.uniform(.4, 1.5)), False, 'NON-CR/NON-PD', 'NE', t, t[2])
    def M():
        t = template(); B = t[1]; v = rng.integers(3); cur = r1(B * rng.uniform(.4, .65))
        if v == 0: return mk('M_incomplete_evidence', 'main', [B], cur, False, None, 'NE', t, t[2])
        if v == 1: return mk('M_incomplete_evidence', 'main', [B], None, False, 'NON-CR/NON-PD', 'NE', t, t[2])
        return mk('M_incomplete_evidence', 'main', [B], cur, False, 'NON-CR/NON-PD', 'NE', t, t[2], lesions_missing=True)
    def N():
        t = template(); B = t[1]; h = [B]
        for _ in range(int(rng.integers(1, 4))): h.append(r1(h[-1] * rng.uniform(.8, 1.12)))
        cur = r1(h[-1] * rng.uniform(.75, 1.4)); return mk('N_repeated_followup', 'main', h, cur, False, 'NON-CR/NON-PD', None, t, t[2])
    for k, fn in ((150, A), (150, Bf), (150, C), (200, D), (100, E), (100, F), (150, G), (200, H), (100, I), (200, J), (150, K), (100, L), (150, M), (100, N)):
        add(k, fn)
    return cases


def build_stress():
    cases, sid = [], 0
    def series(fam, ctype, hist, curs, ref_switch_note):
        nonlocal sid
        sid += 1
        for j, cur in enumerate(curs):
            c = mk(ctype, 'threshold_stress', list(hist), r1(cur), False, 'NON-CR/NON-PD', None, None, 1); c.update(series_id=f'{fam}{sid}', series_pos=j, stress_family=fam); cases.append(c)
    for _ in range(40):                                              # S1: PR boundary at -30% vs baseline (integer baselines)
        B = int(rng.integers(30, 200)); t = 0.7 * B
        series('S1_PR', 'S1_PR_boundary', [B], [t - .2, t - .1, t, t + .1, t + .2], 'PR->SD')
    for _ in range(40):                                              # S2: PD boundary at +20% vs nadir, nadir multiple of 5, abs >= 5 satisfied
        N = int(rng.integers(6, 40)) * 5; t = 1.2 * N
        series('S2_PD', 'S2_PD_boundary_pct', [N], [t - .2, t - .1, t, t + .1, t + .2], 'SD->PD')
    for _ in range(20):                                              # S3: absolute >=5 mm boundary (percentage already >=20%)
        N = int(rng.integers(10, 25)); series('S3_ABS', 'S3_PD_boundary_abs5mm', [N], [N + 4.8, N + 4.9, N + 5.0, N + 5.1, N + 5.2], 'SD->PD')
    for _ in range(25):                                              # S4: new-lesion override (matched pair)
        B = int(rng.integers(30, 200)); cur = r1(B * rng.uniform(.4, .65)); sid += 1
        for j, new in enumerate((False, True)):
            c = mk('S4_new_lesion_override', 'threshold_stress', [B], cur, new, 'NON-CR/NON-PD', None, None, 1); c.update(series_id=f'S4_{sid}', series_pos=j, stress_family='S4_NEW'); cases.append(c)
    for _ in range(25):                                              # S5: nadir dependence (matched pair: same baseline/current, different history)
        B = int(rng.integers(60, 200)); nad = r1(B * rng.uniform(.5, .7)); cur = r1(nad * rng.uniform(1.25, 1.5)); sid += 1
        for j, h in enumerate(([B], [B, nad])):
            c = mk('S5_nadir_dependence', 'threshold_stress', h, cur, False, 'NON-CR/NON-PD', None, None, 1); c.update(series_id=f'S5_{sid}', series_pos=j, stress_family='S5_NADIR'); cases.append(c)
    return cases


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = build_main() + build_stress()
    rows, dis = [], []
    for i, c in enumerate(cases):
        c['case_id'] = f'RC{i:05d}'
        exp, ids, abst = RL.spec_response(c)
        eng = RL.engine_response(c)
        d = RL.derived(c)
        c.update(expected_response=exp, expected_rule_ids=';'.join(ids), expected_abstention=abst, engine_response=eng['response'], engine_rule_ids=';'.join(eng['rule_ids']))
        if eng['response'] != exp or set(eng['rule_ids']) != set(ids):
            dis.append(dict(case_id=c['case_id'], challenge_type=c['challenge_type'], history=json.dumps(c['history']), current_sum=c['current_sum'], new=c['new_lesions'], nt=c['non_target'],
                            spec=exp, engine=eng['response'], spec_ids=';'.join(ids), engine_ids=';'.join(eng['rule_ids'])))
        if c.get('design_expected') and c['design_expected'] != exp:
            c['design_disagrees_with_spec'] = True
        pcts = [abs(d['change_baseline_pct'] + 30) if d else np.nan, abs(d['change_nadir_pct'] - 20) if d else np.nan]
        c['distance_to_threshold'] = float(np.nanmin(pcts)) if d and not np.isnan(pcts).all() else np.nan
        rows.append(dict(case_id=c['case_id'], template_source=c['template_source'], synthetic_flag=True, subset=c['family'], challenge_type=c['challenge_type'],
                         baseline_sum=d.get('baseline_sum'), nadir_sum=d.get('nadir_sum'), current_sum=c['current_sum'], change_baseline_pct=d.get('change_baseline_pct'),
                         change_nadir_pct=d.get('change_nadir_pct'), absolute_change_mm=d.get('absolute_change_from_nadir_mm'), new_lesion=c['new_lesions'],
                         non_target_status=c['non_target'], n_target_lesions=c['n_target_lesions'], history_json=json.dumps(c['history']), all_disappeared=c['all_disappeared'],
                         nodes_lt10=c['nodes_lt10'], lesions_missing=c['lesions_missing'],
                         evidence_complete=not (c['history'] is None or c['current_sum'] is None or c['lesions_missing'] or c['non_target'] is None or c['new_lesions'] is None),
                         expected_response=exp, expected_rule_ids=';'.join(ids), expected_abstention=abst, distance_to_threshold=c['distance_to_threshold'],
                         design_expected=c.get('design_expected'), series_id=c.get('series_id'), series_pos=c.get('series_pos'), stress_family=c.get('stress_family')))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'rule_challenge_cases.csv', index=False)
    pd.DataFrame(dis).to_csv(OUT / 'engine_vs_spec_disagreements.csv', index=False)
    bad_design = [c['case_id'] for c in cases if c.get('design_disagrees_with_spec')]
    print('cases', len(df), 'main', int((df.subset == 'main').sum()), 'stress', int((df.subset == 'threshold_stress').sum()))
    print(df.groupby('challenge_type').size().to_string()); print(df.expected_response.value_counts().to_dict())
    print('engine vs spec disagreements:', len(dis), '| design-intent vs spec disagreements:', len(bad_design))
    if dis: print(pd.DataFrame(dis).head(12).to_string())


if __name__ == '__main__':
    main()
