"""Shared adaptation for running BOTH agents (stateless and generic stateful) on the local Qwen3.5-9B server. Applied identically to both.

The archived exp22b_agent.parse_json slices from the first '{' to the last '}', which fails when a model repeats its JSON answer
(observed with Qwen3.5-9B: correct answer printed twice). parse_json_last returns the LAST complete JSON object in the reply instead.
This is the only behavioural change and it applies to both agents; validation (exp22b_agent.validate) is unchanged.
"""
import json

import exp22b_agent as G


def parse_json_last(text):
    dec, last, i = json.JSONDecoder(), None, 0
    while True:
        i = text.find('{', i)
        if i < 0:
            return last
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                last = obj
            i = end
        except ValueError:
            i += 1


_dumps = json.dumps


def _dumps_safe(o, *a, **k):
    """Harness fix (both agents): exp22b_agent.Toolbox.run json-dumps the tool result outside its try block, so a calculator result that
    is not JSON-serialisable (e.g. the expression 'sum' evaluates to the builtin) crashed the whole episode. Such values are serialised as
    text; serialisable values are unaffected. Installed once at start-up (no per-call swapping, thread-safe)."""
    k.setdefault('default', str)
    return _dumps(o, *a, **k)


def apply():
    G.parse_json = parse_json_last
    G.json.dumps = _dumps_safe
    import pickle
    from pathlib import Path
    f = Path(__file__).resolve().parents[3] / 'outputs/stateful_baseline/tool_cache.pkl'   # pre-computed deterministic tool results
    if f.exists():
        G._cache.update(pickle.loads(f.read_bytes()))
