"""The junction-tree engine: the full conformance suite at exact tolerance,
plus the sizes the exact engine cannot reach."""
import unittest

from carroll_markets.jtree import JunctionTreeEngine, TooTangled
from carroll_markets.oracle import chain, dense, ring, shared, star, tower
from test_engine import EngineConformance


class TestJunctionTree(EngineConformance, unittest.TestCase):
    tol = 1e-9

    def make(self, points, nands, b=100.0):
        return JunctionTreeEngine(points, nands, b)

    def test_width_of_the_tree_like_families_is_two(self):
        for g in (star(40), tower(40), chain(40), shared(40)):
            self.assertEqual(JunctionTreeEngine(*g).width, 2)

    def test_a_ring_of_arguments_is_a_loop_and_stays_narrow(self):
        self.assertLessEqual(JunctionTreeEngine(*ring(60)).width, 4)

    def test_three_hundred_points(self):
        eng = JunctionTreeEngine(*star(150))
        self.assertEqual(len(eng.points), 301)
        eng.trade({'A': 500.0})
        eng.trade({'B7': 300.0, 'R7': 300.0})
        pr = eng.prices()
        self.assertLess(pr['A'], 0.9)
        self.assertGreater(pr['B7'], pr['B8'])
        self.assertAlmostEqual(eng.trade({'B9': 40.0}) + eng.trade({'B9': -40.0}), 0.0, delta=1e-9)

    def test_large_weights_do_not_overflow(self):
        eng = JunctionTreeEngine(*star(100), b=1.0)
        for j in range(100):
            eng.trade({f'B{j}': 600.0})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in eng.prices().values()))

    def test_refuses_a_graph_too_tangled_to_price_exactly(self):
        with self.assertRaises(TooTangled):
            JunctionTreeEngine(*dense(60, 400, seed=1))


if __name__ == '__main__':
    unittest.main()
