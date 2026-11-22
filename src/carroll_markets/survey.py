"""Where exact pricing stops being possible.

    python3 -m carroll_markets.survey            # tables to stdout, results to bench_results/survey_jtree.json

Part 1 times the junction-tree engine on large graphs of each family.
Part 2 sweeps governance-shaped boards: as counterpoints are shared between
more policies, how wide do the tables get (width = largest group of mutually
tied points, minus one; a table has 2^(width+1) entries)?
"""
from __future__ import annotations

import json
import pathlib
import statistics
import time

from .jtree import JunctionTreeEngine, TooTangled, width_of
from .oracle import board, chain, ring, shared, star, tower


def _time(fn, min_seconds=0.15):
    n, t0 = 0, time.perf_counter()
    while True:
        fn()
        n += 1
        dt = time.perf_counter() - t0
        if dt >= min_seconds:
            return dt / n


def timing(name, graph):
    pts, nands = graph
    t0 = time.perf_counter()
    eng = JunctionTreeEngine(pts, nands, 100.0)
    build = time.perf_counter() - t0
    p = pts[len(pts) // 2]

    def read():
        eng._prices = None
        eng.prices()

    return dict(graph=name, points=len(pts), nands=len(nands), width=eng.width,
                table_entries=eng.table_entries, build_s=build,
                trade_s=_time(lambda: (eng.trade({p: 1.0}), eng.trade({p: -1.0}))) / 2,
                prices_s=_time(read))


def main():
    part1 = [timing(n, g) for n, g in (
        ('star(50)', star(50)), ('star(150)', star(150)), ('star(500)', star(500)),
        ('tower(150)', tower(150)), ('chain(150)', chain(150)), ('shared(150)', shared(150)),
        ('ring(100)', ring(100)), ('ring(300)', ring(300)))]
    print('Part 1: junction-tree engine on large graphs')
    print(f"{'graph':>12} {'points':>7} {'width':>6} {'trade ms':>9} {'all prices ms':>14}")
    for r in part1:
        print(f"{r['graph']:>12} {r['points']:>7} {r['width']:>6} {1e3 * r['trade_s']:>9.3f} {1e3 * r['prices_s']:>14.2f}")

    print('\nPart 2: boards with shared counterpoints (10 random boards per row)')
    print(f"{'policies':>9} {'counterpts':>11} {'each bears on':>14} {'points':>7} {'width: median':>14} {'max':>5} {'priced exactly':>15}")
    part2 = []
    for n_pol, n_cp in ((10, 10), (10, 20), (20, 20), (20, 40), (40, 40), (40, 80)):
        for attach in (1, 2, 3, 4):
            widths, ok, npts = [], 0, 0
            for seed in range(10):
                g = board(n_pol, n_cp, attach, n_mitigations=n_cp // 4, seed=seed)
                npts = len(g[0])
                w = width_of(*g)
                widths.append(w)
                ok += w <= JunctionTreeEngine.MAX_WIDTH
            row = dict(policies=n_pol, counterpoints=n_cp, attach=attach, points=npts,
                       width_median=statistics.median(widths), width_max=max(widths), exact_fraction=ok / 10)
            part2.append(row)
            print(f"{n_pol:>9} {n_cp:>11} {attach:>14} {npts:>7} {row['width_median']:>14} {row['width_max']:>5} {ok:>12}/10")
    out = pathlib.Path(__file__).parent / 'bench_results' / 'survey_jtree.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(timing=part1, boards=part2), indent=1))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
