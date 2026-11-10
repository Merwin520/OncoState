#!/usr/bin/env python3
"""Pre-compute the deterministic tool results (exp22b_agent.lesions / transform / matching) for every locked patient in parallel
processes and save them as the agents' tool cache. Tool outputs are identical to on-demand computation (same functions); each entry keeps
the measured uncached compute time, which the efficiency analysis uses. Loaded by local_model_patch.apply() for BOTH agents."""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exp22b_agent as G  # noqa: E402

OUT = HERE.parents[2] / 'outputs/stateful_baseline'


def one(pid):
    G.lesions(pid, 'baseline'); G.lesions(pid, 'followup'); G.transform(pid); G.matching(pid)
    return {k: v for k, v in G._cache.items() if k[1] == pid}


if __name__ == '__main__':
    pids = [p['patient_id'] for p in G.C.pairs_all() if p.get('cohort') == 'locked']
    with Pool(48) as pool:
        parts = pool.map(one, pids, chunksize=1)
    cache = {k: v for p in parts for k, v in p.items()}
    (OUT / 'tool_cache.pkl').write_bytes(pickle.dumps(cache))
    print('patients', len(pids), 'cache entries', len(cache))
