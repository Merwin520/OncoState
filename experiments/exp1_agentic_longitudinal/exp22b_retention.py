"""Correction retention in the sequential CT study (Methods: one reference-derived correction event per patient, identical for
every execution condition).

OncoState     : the correction is written to the patient state; the state is saved and re-loaded for a new session, and queries
                Q1-Q3 are answered from it.
Generic agent : the correction is supplied in the Q1 episode; Q2 and Q3 are new episodes, as in the main comparison.
Each answer is scored for whether it reflects the correction.
"""
import json, os, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

import exp22b_common as C
import exp22b_run as RUN
import exp22b_agent as AG

BASE = C.BASE
N_EVENTS, SEED = 60, 20260921


def events(det):
    rng = np.random.default_rng(SEED); ev = []
    for p in RUN.LOCKED:
        r = det.get((p, 'Full OncoAgent'))
        if not r: continue
        ref = r['ref']; pairs = set(r['pred']['pairs'])
        for g, cls in ref['gclass'].items():
            pb, pf = ref['inv_b'].get(g), ref['inv_f'].get(g)
            if cls == 'persistent' and pb is not None and pf is not None and (pb, pf) not in pairs:
                ev.append(dict(pid=p, gt=g, pb=pb, pf=pf, rows=r['rows'], bl=r['bl'], fu=r['fu']))
    idx = rng.permutation(len(ev))[:N_EVENTS]; return [ev[i] for i in sorted(idx)], len(ev)


def relink(rows, pb, pf):
    keep = [dict(x) for x in rows if x['baseline_lesion_id'] != pb and x['followup_lesion_id'] != pf]
    for x in rows:
        if x['status'] == 'persistent' and (x['baseline_lesion_id'] == pb) != (x['followup_lesion_id'] == pf):
            if x['baseline_lesion_id'] != pb: keep.append({'baseline_lesion_id': x['baseline_lesion_id'], 'followup_lesion_id': None, 'status': 'disappearing'})
            if x['followup_lesion_id'] != pf: keep.append({'baseline_lesion_id': None, 'followup_lesion_id': x['followup_lesion_id'], 'status': 'newly_appearing'})
    keep.append({'baseline_lesion_id': pb, 'followup_lesion_id': pf, 'status': 'persistent', 'edited': True}); return keep


def reflects(qid, ans, pb, pf):
    if not ans: return False
    if qid == 'Q1': return any(m['baseline_lesion_id'] == pb and m['followup_lesion_id'] == pf for m in ans['matches'])
    if qid == 'Q2': return ans['baseline_status'].get(str(pb)) == 'persistent' and ans['followup_status'].get(str(pf)) == 'persistent'
    if qid == 'Q3': return any(m['baseline_lesion_id'] == pb and m['followup_lesion_id'] == pf for m in ans['changes'])


def main():
    from openai import OpenAI
    det, _ = RUN.deterministic(); evs, n_all = events(det); rows = []
    # ---- Full: persistent state, edited once, saved, re-loaded, queried
    for e in evs:
        edited = relink(e['rows'], e['pb'], e['pf'])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'state.json'; f.write_text(json.dumps({'correspondences': edited, 'baseline_lesions': {str(k): v for k, v in e['bl'].items()}, 'followup_lesions': {str(k): v for k, v in e['fu'].items()}}))
            st = json.loads(f.read_text())                                              # new session: state read back from disk
        pred = C.pred_from_rows(st['correspondences'], {int(k): v for k, v in st['baseline_lesions'].items()}, {int(k): v for k, v in st['followup_lesions'].items()})
        ans = {'Q1': (e['pb'], e['pf']) in pred['pairs'], 'Q2': pred['status_b'].get(e['pb']) == 'persistent' and pred['status_f'].get(e['pf']) == 'persistent', 'Q3': (e['pb'], e['pf']) in pred['change']}
        for q, ok in ans.items(): rows.append(dict(system='Full OncoAgent', patient_id=e['pid'], pb=e['pb'], pf=e['pf'], query=q, reflects_correction=bool(ok), context='persistent state re-loaded'))
    # generic agent: correction in the Q1 episode only
    if os.environ.get('EXP22B_SKIP_AGENT'):
        df = pd.DataFrame(rows); df.to_csv(BASE / 'edit_retention.csv', index=False)
        summ = df.groupby(['system', 'query']).reflects_correction.agg(['mean', 'sum', 'count']).reset_index(); summ.to_csv(BASE / 'edit_retention_summary.csv', index=False)
        print(f'{len(evs)} correction events (generic agent skipped)'); print(summ.to_string(index=False)); return
    client = OpenAI(); note = lambda e: f"Verified clinician correction (apply it): baseline lesion {e['pb']} and follow-up lesion {e['pf']} are the same lesion, i.e. a matched pair."
    jobs = [(e, q, note(e) if q == 'Q1' else None) for e in evs for q in ('Q1', 'Q2', 'Q3')]
    with ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(AG.episode, client, e['pid'], 0, q, n): (e, q) for e, q, n in jobs}
        for fu in as_completed(futs):
            e, q = futs[fu]; r = fu.result()
            rows.append(dict(system=AG.__name__ and 'Generic state-free agent', patient_id=e['pid'], pb=e['pb'], pf=e['pf'], query=q, reflects_correction=bool(r['success'] and reflects(q, r['structured_answer'], e['pb'], e['pf'])),
                             context='correction in this episode only' if q == 'Q1' else 'fresh episode (no state, no memory of the correction)', success=r['success'], token_count=r['token_count']))
    df = pd.DataFrame(rows); df.to_csv(BASE / 'edit_retention.csv', index=False)
    summ = df.groupby(['system', 'query']).reflects_correction.agg(['mean', 'sum', 'count']).reset_index(); summ.to_csv(BASE / 'edit_retention_summary.csv', index=False)
    print(f'{len(evs)} correction events'); print(summ.to_string(index=False))


if __name__ == '__main__':
    main()
