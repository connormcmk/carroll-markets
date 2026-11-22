"""Belief propagation between exact clusters: the tight pairs that plain
belief propagation gets wrong are priced exactly. Thresholds record what was
measured on 2026-11-16; they are not yet acceptance criteria."""
import unittest

from carroll_markets.bp import BeliefPropagationEngine
from carroll_markets.cbp import ClusterBPEngine, clusters_of
from carroll_markets.jtree import JunctionTreeEngine
from carroll_markets.oracle import argument, board, compare, dense, ring, star, tower, trade_sequence

make = lambda p, n, b=100.0: ClusterBPEngine(p, n, b)


def worst(engine_cls, g, n=40):
    eng, jt = engine_cls(*g), JunctionTreeEngine(*g)
    for d in trade_sequence(g[0], n, seed=1):
        eng.trade(d)
        jt.trade(d)
    pe, pj = eng.prices(), jt.prices()
    return max(abs(pe[p] - pj[p]) for p in g[0])


class TestClusters(unittest.TestCase):
    def test_no_two_clusters_share_two_points(self):
        pts, nands = argument(10, 60, 0.3, seed=4)
        idx = {p: i for i, p in enumerate(pts)}
        tris = [tuple(idx[p] for p in t) for t in nands]
        scopes = [set(v for k in g for v in tris[k]) for g in clusters_of(tris)]
        self.assertEqual(sorted(k for g in clusters_of(tris) for k in g), list(range(len(tris))))
        for i, a in enumerate(scopes):
            for b in scopes[:i]:
                self.assertLessEqual(len(a & b), 1)

    def test_the_size_limit_is_respected(self):
        pts, nands = dense(12, 20, 4)
        idx = {p: i for i, p in enumerate(pts)}
        tris = [tuple(idx[p] for p in t) for t in nands]
        for g in clusters_of(tris, max_cluster=6):
            self.assertLessEqual(len({v for k in g for v in tris[k]}), 6)


class TestClusterBP(unittest.TestCase):
    def test_exact_without_loops_and_identical_to_plain_when_nothing_merges(self):
        for g in (star(4), tower(4)):
            self.assertLess(worst(ClusterBPEngine, g), 1e-9)
        g = ring(5)
        a, b = ClusterBPEngine(*g), BeliefPropagationEngine(*g)
        for d in trade_sequence(g[0], 30, seed=2):
            self.assertAlmostEqual(a.trade(d), b.trade(d), places=8)

    def test_round_trips_are_free_order_does_not_matter_and_nothing_locks_in_a_profit(self):
        for g in (argument(4, 5, 0.6, seed=2), argument(3, 6, 0.5, seed=5), dense(8, 9, 2)):
            r = compare(make, g, n_trades=30, seed=3, profit_tries=80)
            self.assertLess(r['round_trip_leak'], 1e-8)
            self.assertLess(r['order_dependence'], 1e-8)
            self.assertLess(r['guaranteed_profit'], 1e-6)

    def test_tight_pairs_are_priced_exactly(self):
        g = argument(5, 20, 0.2, seed=1)                  # plain belief propagation is off by 0.04 here
        self.assertGreater(worst(BeliefPropagationEngine, g), 0.01)
        self.assertLess(worst(ClusterBPEngine, g), 1e-6)

    def test_graphs_that_are_one_tangle_become_exact(self):
        for g in (dense(6, 5, 1), dense(8, 9, 2)):
            self.assertLess(worst(ClusterBPEngine, g), 1e-8)

    def test_argument_shaped_graphs_stay_within_half_a_cent(self):
        for seed in (0, 3, 4):
            self.assertLess(worst(ClusterBPEngine, argument(10, 60, 0.3, seed=seed)), 0.005)

    def test_boards_are_no_worse_than_plain(self):
        g = board(10, 20, 3, n_mitigations=5, seed=0)
        self.assertLessEqual(worst(ClusterBPEngine, g), worst(BeliefPropagationEngine, g) + 1e-9)


if __name__ == '__main__':
    unittest.main()
