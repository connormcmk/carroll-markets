"""How the exact engine's cost grows with graph size.

    python3 -m carroll_markets.bench                 # table to stdout
    python3 -m carroll_markets.bench --save          # also writes bench_results/baseline_exact.json

One row per graph size: assignments enumerated, seconds to build, and seconds
per price read and per trade. A candidate engine is benchmarked the same way
by passing its factory to `run`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from .engine import ExactEngine
from .oracle import star


def _time(fn, min_seconds=0.2):
    n, t0 = 0, time.perf_counter()
    while True:
        fn()
        n += 1
        dt = time.perf_counter() - t0
        if dt >= min_seconds:
            return dt / n


def run(make_engine=ExactEngine, sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9), budget_seconds=60.0):
    rows = []
    for k in sizes:
        pts, nands = star(k)
        t0 = time.perf_counter()
        eng = make_engine(pts, nands, 100.0)
        build = time.perf_counter() - t0
        eng.trade({'A': 30.0})
        row = dict(points=len(pts), nands=len(nands),
                   assignments=getattr(eng, 'n_outcomes', None),
                   build_s=build,
                   prices_s=_time(eng.prices),
                   trade_s=_time(lambda: (eng.trade({'B0': 1.0}), eng.trade({'B0': -1.0}))) / 2)
        rows.append(row)
        if build + row['prices_s'] > budget_seconds:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--max-k', type=int, default=9, help='largest star: 2k+1 points')
    a = ap.parse_args()
    rows = run(sizes=tuple(range(1, a.max_k + 1)))
    print(f"{'points':>7} {'assignments':>12} {'build s':>9} {'prices s':>10} {'trade s':>10}")
    for r in rows:
        print(f"{r['points']:>7} {str(r['assignments']):>12} {r['build_s']:>9.4f} {r['prices_s']:>10.6f} {r['trade_s']:>10.6f}")
    if len(rows) >= 2:
        g = (rows[-1]['prices_s'] / rows[-2]['prices_s']) ** (1 / (rows[-1]['points'] - rows[-2]['points']))
        print(f"\ncost of a price read multiplies by about {g:.2f} for each point added")
    if a.save:
        out = pathlib.Path(__file__).parent / 'bench_results' / 'baseline_exact.json'
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(dict(engine='ExactEngine', family='star', rows=rows), indent=1))
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
