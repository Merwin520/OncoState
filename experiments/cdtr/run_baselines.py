#!/usr/bin/env python3
"""Comparators under the same evidence, state update, engine and global review budget as CDT-R.
  python run_baselines.py --split policy_validation|test     -> results/<split>/{v0_no_review,random,v0_confidence,set_ambiguity}.json"""
import argparse

from run_cdtr import cfg, run_one

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='policy_validation'); ap.add_argument('--alpha', type=float)
    a = ap.parse_args(); c = cfg()
    for pol in ('v0_no_review', 'v0_confidence', 'set_ambiguity'):
        run_one(pol, a.split, a.alpha)
    run_one('random', a.split, a.alpha, seeds=tuple(c['random_seeds']))
