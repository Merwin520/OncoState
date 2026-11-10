#!/usr/bin/env python3
"""Stateless tool agent (exp22b_agent.main, unchanged) re-run on the local model with the shared parser patch, into a separate raw file.
  EXP22B_MODEL=qwen3.5-9b OPENAI_BASE_URL=http://127.0.0.1:8610/v1 OPENAI_API_KEY=dummy python run_stateless.py [--pilot N] [--workers W]"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import local_model_patch  # noqa: E402
import exp22b_agent as G  # noqa: E402
local_model_patch.apply()
OUT = HERE.parents[2] / 'outputs/stateful_baseline'
if '--raw' not in sys.argv:
    sys.argv += ['--raw', str(OUT / 'stateless_raw.jsonl')]
G.main()
