"""Loopy belief propagation: exact where there are no loops, and on loops an
approximation whose round trips are free and which offers no free money.
The error thresholds here record what was measured on 2026-11-15; they are
not yet acceptance criteria."""
import unittest

from carroll_markets.bp import BeliefPropagationEngine
from carroll_markets.engine import ExactEngine
from carroll_markets.jtree import JunctionTreeEngine
from carroll_markets.oracle import argument, board, chain, compare, ring, shared, star, tower, trade_sequence

make = lambda p, n, b=100.0: BeliefPropagationEngine(p, n, b)


class TestBeliefPropagation(unittest.TestCase):
    def test_exact_on_graphs_without_loops(self):
        for g in (star(4), tower(4), chain(4), shared(4)):
            r = compare(make, g, n_trades=40, seed=3, profit_tries=20)
            self.assertLess(r['max_price_error'], 1e-9)
            self.assertLess(r['max_cost_error'], 1e-7)

    def test_on_loops_round_trips_are_free_and_order_does_not_matter(self):
        for g in (ring(3), ring(5), board(3, 3, 2, 1, seed=1)):
            r = compare(make, g, n_trades=40, seed=3, profit_tries=20)
            self.assertLess(r['round_trip_leak'], 1e-8)
            self.assertLess(r['order_dependence'], 1e-8)

    def test_on_loops_no_trade_locks_in_a_profit(self):
        for g in (ring(3), ring(5), board(3, 3, 2, 1, seed=1), board(3, 3, 2, 1, seed=2)):
            self.assertLess(compare(make, g, n_trades=25, seed=3, profit_tries=120)['guaranteed_profit'], 1e-6)

    def test_price_error_on_a_board_with_shared_counterpoints(self):
        # 110 points, each counterpoint on 3 of 20 policies; the junction tree is the exact reference
        g = board(20, 20, 3, n_mitigations=5, seed=0)
        bp, jt = BeliefPropagationEngine(*g), JunctionTreeEngine(*g)
        for d in trade_sequence(g[0], 40, seed=1):
            bp.trade(d)
            jt.trade(d)
        pb, pj = bp.prices(), jt.prices()
        self.assertLess(max(abs(pb[p] - pj[p]) for p in g[0]), 0.005)

    def test_argument_shaped_graphs_deep_is_exact_and_tight_pairs_are_where_it_errs(self):
        def worst(g):
            bp, jt = BeliefPropagationEngine(*g), JunctionTreeEngine(*g)
            for d in trade_sequence(g[0], 40, seed=1):
                bp.trade(d)
                jt.trade(d)
            pb, pj = bp.prices(), jt.prices()
            return max(abs(pb[p] - pj[p]) for p in g[0])

        def tight_pairs(g):
            n = g[1]
            return sum(1 for i in range(len(n)) for j in range(i) if len(set(n[i]) & set(n[j])) >= 2)

        deep = argument(5, 20, reuse=0.0, seed=0)                  # mitigations of mitigations, no loops
        self.assertLess(worst(deep), 1e-8)
        clean, tight = argument(5, 20, 0.2, seed=3), argument(5, 20, 0.2, seed=1)
        self.assertEqual(tight_pairs(clean), 0)
        self.assertGreater(tight_pairs(tight), 0)
        self.assertLess(worst(clean), 0.003)
        self.assertGreater(worst(tight), 0.01)                     # the weakness, recorded until it is fixed
        self.assertLess(worst(tight), 0.06)

    def test_settlement_on_a_tree_matches_exact(self):
        g = star(3)
        bp, ex = BeliefPropagationEngine(*g), ExactEngine(*g)
        for d in trade_sequence(g[0], 20, seed=4):
            bp.trade(d)
            ex.trade(d)
        bp.settle('B1', True)
        ex.settle('B1', True)
        for p in g[0]:
            self.assertAlmostEqual(bp.price(p), ex.price(p), delta=1e-8)


if __name__ == '__main__':
    unittest.main()
