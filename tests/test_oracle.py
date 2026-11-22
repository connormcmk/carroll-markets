"""The oracle's measures must be able to tell a bad engine from a good one."""
import math
import unittest

from carroll_markets.engine import Engine, ExactEngine
from carroll_markets.oracle import argument, compare, guaranteed_profit, star


class Unlinked(Engine):
    """A deliberately wrong engine: every point priced alone, NANDs ignored."""

    def __init__(self, points, nands, b=100.0):
        self.points, self.nands, self.b = list(points), list(nands), b
        self.s = {p: 0.0 for p in points}

    def _c(self, s):
        return self.b * sum(math.log1p(math.exp(v / self.b)) for v in s.values())

    def prices(self):
        return {p: 1 / (1 + math.exp(-v / self.b)) for p, v in self.s.items()}

    def cost(self, delta):
        s = dict(self.s)
        for p, d in delta.items():
            s[p] += d
        return self._c(s) - self._c(self.s)

    def trade(self, delta):
        paid = self.cost(delta)
        for p, d in delta.items():
            self.s[p] += d
        return paid


class TestOracleHasTeeth(unittest.TestCase):
    def test_exact_engine_scores_clean(self):
        r = compare(lambda p, n, b: ExactEngine(p, n, b), star(3))
        self.assertLess(r['max_price_error'], 1e-12)
        self.assertLess(r['guaranteed_profit'], 1e-7)
        self.assertLess(r['informed_profit'], 1e-6)

    def test_a_wrong_engine_is_caught_on_accuracy_and_on_informed_profit(self):
        r = compare(lambda p, n, b: Unlinked(p, n, b), star(3))
        self.assertGreater(r['max_price_error'], 0.02)
        self.assertGreater(r['informed_profit'], 0.1)
        self.assertLess(r['round_trip_leak'], 1e-9)        # it is still a cost function

    def test_free_money_is_found_when_it_exists(self):
        # push all three of a NAND triple high in the unlinked engine: selling
        # one of each then collects more than the 2 it can ever owe
        pts, nands = ['A', 'B', 'R'], [('A', 'B', 'R')]
        bad, exact = Unlinked(pts, nands), ExactEngine(pts, nands)
        bad.trade({'A': 300.0, 'B': 300.0, 'R': 300.0})
        self.assertGreater(guaranteed_profit(bad, exact, tries=400), 1.0)
        good = ExactEngine(pts, nands)
        good.trade({'A': 300.0, 'B': 300.0, 'R': 300.0})
        self.assertLess(guaranteed_profit(good, exact, tries=400), 1e-7)


class TestArgumentFamily(unittest.TestCase):
    def test_it_has_deep_mitigations_and_points_in_two_roles(self):
        pts, nands = argument(10, 60, reuse=0.3, seed=0)
        self.assertEqual(len(set(pts)), len(pts))
        rel_of = {r: a for a, _, r in nands}

        def depth(r):
            return 1 + (depth(rel_of[r]) if rel_of[r] in rel_of else 0)

        self.assertGreaterEqual(max(depth(r) for r in rel_of), 3)          # mitigations of mitigations
        roles = {}
        for a, c, _ in nands:
            roles.setdefault(c, set()).add('mitigation' if a in rel_of else 'counterpoint')
        self.assertTrue(any(len(v) == 2 for v in roles.values()))          # one point, both roles

    def test_without_reuse_it_has_no_loops(self):
        from carroll_markets.jtree import width_of
        self.assertEqual(width_of(*argument(8, 30, reuse=0.0, seed=1)), 2)


if __name__ == '__main__':
    unittest.main()
