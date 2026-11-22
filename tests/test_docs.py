"""The numbers quoted in README.md and docs/ are checked here, so the
documents cannot drift from the code."""
import importlib.util
import pathlib
import unittest

from carroll_markets import ExactEngine, JunctionTreeEngine

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestReadmeExample(unittest.TestCase):
    def test_the_minimal_example(self):
        for cls in (JunctionTreeEngine, ExactEngine):
            m = cls(['claim', 'counterpoint', 'relevance'], [('claim', 'counterpoint', 'relevance')], b=100.0)
            for p in m.prices().values():
                self.assertAlmostEqual(p, 3 / 7, places=9)
            self.assertAlmostEqual(m.trade({'counterpoint': 80.0}), 42.21, delta=0.005)
            pr = m.prices()
            self.assertAlmostEqual(pr['counterpoint'], 0.625, delta=0.0005)
            self.assertAlmostEqual(pr['claim'], 0.396, delta=0.0005)
            self.assertAlmostEqual(pr['relevance'], 0.396, delta=0.0005)
            self.assertAlmostEqual(m.trade({'relevance': 80.0}), 39.54, delta=0.005)
            pr = m.prices()
            self.assertAlmostEqual(pr['counterpoint'], 0.593, delta=0.0005)
            self.assertAlmostEqual(pr['relevance'], 0.593, delta=0.0005)
            self.assertAlmostEqual(pr['claim'], 0.344, delta=0.0005)


class TestWorkedMarket(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('worked_market', ROOT / 'examples' / 'worked_market.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.steps = [prices for _, prices in mod.run(verbose=False)]

    def test_the_story_told_in_the_mechanism_document(self):
        opening, claim_bought, cp_bought, rel_backed, second_cp, settled = self.steps
        self.assertAlmostEqual(opening['reopen'], 0.5, places=9)
        self.assertAlmostEqual(claim_bought['reopen'], 0.769, delta=0.0005)
        # a counterpoint with no one behind its relevance barely touches the claim
        self.assertAlmostEqual(cp_bought['crack'], 0.82, delta=0.005)
        self.assertAlmostEqual(cp_bought['reopen'], 0.767, delta=0.0005)
        self.assertLess(cp_bought['crack_matters'], 0.01)
        # backing the relevance is what moves it
        self.assertAlmostEqual(rel_backed['crack_matters'], 0.45, delta=0.005)
        self.assertAlmostEqual(rel_backed['reopen'], 0.591, delta=0.0005)
        # the second counterpoint, unbacked, again does almost nothing
        self.assertAlmostEqual(second_cp['budget'], 0.82, delta=0.005)
        self.assertAlmostEqual(second_cp['reopen'], 0.589, delta=0.0005)
        self.assertLess(rel_backed['reopen'] - second_cp['reopen'], 0.003)
        # settling the first counterpoint false releases the claim
        self.assertAlmostEqual(settled['reopen'], 0.767, delta=0.0005)
        self.assertEqual(settled['crack'], 0.0)


if __name__ == '__main__':
    unittest.main()
