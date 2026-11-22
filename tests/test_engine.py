"""What every pricing engine must satisfy.

`EngineConformance` holds the checks; a candidate engine gets the whole suite
by subclassing it with its own `make` and tolerance:

    class TestMine(EngineConformance, unittest.TestCase):
        tol = 1e-3
        def make(self, points, nands, b=100.0): return MyEngine(points, nands, b)

The checks compare against ExactEngine wherever a right answer exists, and
otherwise assert properties that hold for any sound market maker.
"""
import itertools
import math
import unittest

from carroll_markets.engine import ExactEngine
from carroll_markets.oracle import chain, dense, ring, shared, star, tower, trade_sequence

GRAPHS = [('star3', star(3)), ('tower3', tower(3)), ('chain3', chain(3)),
          ('shared3', shared(3)), ('ring4', ring(4)), ('dense8', dense(8, 9, 2))]


class EngineConformance:
    tol = 1e-9

    def make(self, points, nands, b=100.0):
        raise NotImplementedError

    def loaded(self, graph, seed=0, n=25, b=100.0):
        pts, nands = graph
        eng, ref = self.make(list(pts), list(nands), b), ExactEngine(pts, nands, b)
        for delta in trade_sequence(pts, n, seed=seed):
            eng.trade(delta)
            ref.trade(delta)
        return eng, ref

    def test_empty_market_prices_match_exact(self):
        for name, (pts, nands) in GRAPHS:
            eng, ref = self.make(list(pts), list(nands)), ExactEngine(pts, nands)
            for p in pts:
                self.assertAlmostEqual(eng.price(p), ref.price(p), delta=self.tol, msg=f'{name} {p}')

    def test_prices_and_costs_match_exact_along_a_trade_path(self):
        for name, g in GRAPHS:
            pts, nands = g
            eng, ref = self.make(list(pts), list(nands)), ExactEngine(pts, nands)
            for step, delta in enumerate(trade_sequence(pts, 40, seed=7)):
                paid, want = eng.trade(delta), ref.trade(delta)
                self.assertAlmostEqual(paid, want, delta=self.tol * max(1.0, abs(want)), msg=f'{name} step {step}')
                pe, pr = eng.prices(), ref.prices()
                for p in pts:
                    self.assertAlmostEqual(pe[p], pr[p], delta=self.tol, msg=f'{name} step {step} {p}')

    def test_prices_are_probabilities(self):
        for name, g in GRAPHS:
            eng, _ = self.loaded(g)
            for p, v in eng.prices().items():
                self.assertGreater(v, 0.0, f'{name} {p}')
                self.assertLess(v, 1.0, f'{name} {p}')

    def test_no_nand_triple_can_be_jointly_certain(self):
        # no admissible assignment has all three true, so the three prices sum to at most 2
        for name, g in GRAPHS:
            eng, _ = self.loaded(g, seed=3, n=60)
            pr = eng.prices()
            for a, b, r in g[1]:
                self.assertLessEqual(pr[a] + pr[b] + pr[r], 2.0 + self.tol, f'{name} {a},{b},{r}')

    def test_round_trip_is_free(self):
        for name, g in GRAPHS:
            eng, _ = self.loaded(g)
            for p in g[0][:4]:
                paid = eng.trade({p: 17.0}) + eng.trade({p: -17.0})
                self.assertAlmostEqual(paid, 0.0, delta=self.tol, msg=f'{name} {p}')

    def test_cost_does_not_depend_on_order(self):
        for name, (pts, nands) in GRAPHS:
            seq = trade_sequence(pts, 12, seed=11)
            totals = []
            for order in (seq, list(reversed(seq)), seq[1::2] + seq[0::2]):
                eng = self.make(list(pts), list(nands))
                totals.append(sum(eng.trade(d) for d in order))
            self.assertAlmostEqual(totals[0], totals[1], delta=self.tol * 10, msg=name)
            self.assertAlmostEqual(totals[0], totals[2], delta=self.tol * 10, msg=name)

    def test_quote_equals_charge(self):
        for name, g in GRAPHS:
            eng, _ = self.loaded(g)
            p = g[0][0]
            quoted = eng.cost({p: 9.0})
            self.assertAlmostEqual(quoted, eng.trade({p: 9.0}), delta=self.tol, msg=name)

    def test_buying_raises_the_price_and_costs_more_each_share(self):
        for name, g in GRAPHS:
            eng, _ = self.loaded(g)
            p = g[0][1]
            before = eng.price(p)
            first = eng.trade({p: 10.0})
            second = eng.trade({p: 10.0})
            self.assertGreater(eng.price(p), before, name)
            self.assertGreater(second, first - self.tol, name)

    def test_price_is_the_marginal_cost(self):
        for name, g in GRAPHS:
            eng, _ = self.loaded(g)
            h = 1e-4
            for p in g[0][:3]:
                slope = (eng.cost({p: h}) - eng.cost({p: -h})) / (2 * h)
                self.assertAlmostEqual(slope, eng.price(p), delta=max(self.tol, 1e-6), msg=f'{name} {p}')

    def test_the_argument_moves_the_claim(self):
        # the counterpoint and its relevance, both bought, push the claim down;
        # the counterpoint alone, with relevance sold off, barely touches it
        both = self.make(['A', 'B', 'R'], [('A', 'B', 'R')])
        a0 = both.price('A')
        both.trade({'B': 150.0, 'R': 150.0})
        self.assertLess(both.price('A'), a0 - 0.15)
        tortoise = self.make(['A', 'B', 'R'], [('A', 'B', 'R')])
        tortoise.trade({'B': 150.0})
        tortoise.trade({'R': -150.0})
        self.assertLess(abs(tortoise.price('A') - 0.5), 0.06)

    def test_settlement_matches_exact(self):
        for name, g in GRAPHS:
            eng, ref = self.loaded(g, seed=5)
            p = g[0][1]
            eng.settle(p, True)
            ref.settle(p, True)
            self.assertAlmostEqual(eng.price(p), 1.0, delta=self.tol, msg=name)
            for q in g[0]:
                self.assertAlmostEqual(eng.price(q), ref.price(q), delta=self.tol, msg=f'{name} {q}')

    def test_settling_counterpoint_and_relevance_true_kills_the_claim(self):
        eng = self.make(['A', 'B', 'R'], [('A', 'B', 'R')])
        eng.trade({'A': 80.0})
        eng.settle('B', True)
        eng.settle('R', True)
        self.assertAlmostEqual(eng.price('A'), 0.0, delta=self.tol)

    def test_offers_no_free_money(self):
        from carroll_markets.oracle import guaranteed_profit, informed_profit
        for name, g in GRAPHS:
            eng, ref = self.loaded(g, seed=2, n=40)
            self.assertLessEqual(guaranteed_profit(eng, ref, tries=120), max(self.tol, 1e-7), name)
            self.assertLessEqual(informed_profit(eng, ref), max(self.tol * 100, 1e-6), name)

    # ---- growing a live market ----

    def priced(self, pa, pb, b=100.0):
        eng = self.make(['A', 'B'], [], b)
        for p, want in (('A', pa), ('B', pb)):
            if want != 0.5:
                eng.trade({p: b * math.log(want / (1 - want))})
        return eng

    def test_adding_an_unattached_point_moves_no_price(self):
        for name, g in GRAPHS:
            eng, _ = self.loaded(g)
            before = eng.prices()
            eng.add_point('NEW')
            after = eng.prices()
            self.assertAlmostEqual(after['NEW'], 0.5, delta=self.tol, msg=name)
            for p in g[0]:
                self.assertAlmostEqual(after[p], before[p], delta=self.tol, msg=f'{name} {p}')

    def test_a_new_relevance_enters_low_and_moves_almost_nothing(self):
        for pa, pb in ((0.5, 0.5), (0.9, 0.5), (0.5, 0.9), (0.9, 0.9), (0.2, 0.7)):
            eng = self.priced(pa, pb)
            eng.add_relevance('A', 'B', 'R', entry_price=0.01)
            pr = eng.prices()
            self.assertAlmostEqual(pr['A'], pa, delta=0.003, msg=f'{pa},{pb}')
            self.assertAlmostEqual(pr['B'], pb, delta=0.003, msg=f'{pa},{pb}')
            self.assertLessEqual(pr['R'], 0.01 + self.tol)
            eng = self.priced(pa, pb)                       # the lower it enters, the less it moves
            eng.add_relevance('A', 'B', 'R', entry_price=1e-6)
            pr = eng.prices()
            self.assertAlmostEqual(pr['A'], pa, delta=max(self.tol, 1e-6), msg=f'{pa},{pb}')
            self.assertAlmostEqual(pr['B'], pb, delta=max(self.tol, 1e-6), msg=f'{pa},{pb}')

    def test_a_new_relevance_moves_no_other_point_to_speak_of(self):
        eng, _ = self.loaded(star(3))                      # B0 and B1 are already linked through A
        before = eng.prices()
        eng.add_relevance('B0', 'B1', 'RX', entry_price=0.01)
        after = eng.prices()
        for p in before:
            self.assertAlmostEqual(after[p], before[p], delta=0.003, msg=p)

    def test_raising_a_relevance_costs_the_same_whatever_stands_behind_it(self):
        # moving a price from p0 to p1 costs b*ln((1-p0)/(1-p1)), for any point in any market
        for pa, pb in ((0.9, 0.5), (0.5, 0.9), (0.9, 0.9), (0.5, 0.5)):
            eng = self.priced(pa, pb)
            eng.add_relevance('A', 'B', 'R')
            p0 = eng.price('R')
            lo, hi = 0.0, 5000.0
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                eng.trade({'R': mid})
                reached = eng.price('R')
                eng.trade({'R': -mid})
                lo, hi = (mid, hi) if reached < 0.60 else (lo, mid)
            want = 100.0 * math.log((1.0 - p0) / 0.4)
            self.assertAlmostEqual(eng.trade({'R': hi}), want, delta=max(self.tol * 100, 1e-4), msg=f'{pa},{pb}')

    def test_when_relevance_rises_the_point_nearer_one_half_gives_way_most(self):
        eng = self.priced(0.9, 0.5)                         # claim strongly held, counterpoint not
        eng.add_relevance('A', 'B', 'R')
        eng.trade({'R': 560.0})                             # takes the relevance to about 0.6
        pr = eng.prices()
        self.assertAlmostEqual(pr['R'], 0.6, delta=0.05)
        self.assertGreater(0.5 - pr['B'], 3 * (0.9 - pr['A']))     # the counterpoint falls much further
        self.assertLess(pr['A'], 0.9 - 0.02)                       # but the claim is not immune
        self.assertGreater(pr['B'], 0.05)                          # and the counterpoint does not collapse

    def test_joint_probability_read_from_the_cost_function(self):
        for name, g in GRAPHS:
            eng, exact = self.loaded(g, seed=5, n=30)
            worlds = exact.admissible()
            a, b = g[0][0], g[0][-1]
            f = exact.outstanding()                # brute force over the exact engine's assignments
            w = [math.exp(sum(f[p] * x[p] for p in g[0]) / exact.b) for x in worlds]
            want = sum(wi for wi, x in zip(w, worlds) if x[a] and x[b]) / sum(w)
            self.assertAlmostEqual(eng.joint_true(a, b), want, delta=max(self.tol, 1e-7), msg=name)

    def test_a_relevance_can_be_made_to_land_on_a_named_price(self):
        for pa, pb in ((0.9, 0.5), (0.9, 0.9), (0.2, 0.7)):
            eng = self.priced(pa, pb)
            eng.add_relevance('A', 'B', 'R', entry_price=0.10, landed=True)
            self.assertAlmostEqual(eng.price('R'), 0.10, delta=max(self.tol, 1e-7))

    def test_shares_for_spend_inverts_cost(self):
        eng, _ = self.loaded(GRAPHS[0][1])
        n = eng.shares_for_spend('A', 25.0)
        self.assertAlmostEqual(eng.cost({'A': n}), 25.0, delta=1e-6)


class TestExactEngine(EngineConformance, unittest.TestCase):
    def make(self, points, nands, b=100.0):
        return ExactEngine(points, nands, b)

    def test_agrees_with_an_independent_recomputation(self):
        for name, g in GRAPHS:
            eng, _ = self.loaded(g, seed=9, n=50)
            prices, logz = eng.reference_prices_and_logz()
            self.assertAlmostEqual(logz, eng.log_partition(), delta=1e-9, msg=name)
            for p, v in eng.prices().items():
                self.assertAlmostEqual(v, prices[p], delta=1e-12, msg=f'{name} {p}')

    def test_assignment_count(self):
        # one NAND removes one of eight assignments of its three points
        self.assertEqual(ExactEngine(['A', 'B', 'R'], [('A', 'B', 'R')]).n_outcomes, 7)
        # a star with k counterpoints: A false -> 4^k, A true -> 3^k
        for k in (1, 2, 3, 4):
            self.assertEqual(ExactEngine(*star(k)).n_outcomes, 4 ** k + 3 ** k)

    def test_market_maker_loss_is_bounded(self):
        # whatever is traded and however the world turns out, the maker loses
        # at most b * log(number of admissible assignments)
        pts, nands = star(3)
        eng = ExactEngine(pts, nands, b=50.0)
        collected = sum(eng.trade(d) for d in trade_sequence(pts, 80, seed=4, size=200.0))
        out = eng.outstanding()
        bound = eng.b * math.log(eng.n_outcomes)
        for bits in itertools.product([0, 1], repeat=len(pts)):
            x = dict(zip(pts, bits))
            if any(x[a] and x[b] and x[r] for a, b, r in nands):
                continue
            payout = sum(out[p] for p in pts if x[p])
            self.assertLessEqual(payout - collected, bound + 1e-6)

    def test_golden_single_argument(self):
        # b=120; spend 60 on A, then 25 on B, then 35 on R. A fixed book kept as
        # a regression value; an independent implementation gives the same
        # A=0.60 B=0.46 R=0.52.
        eng = ExactEngine(['A', 'B', 'R'], [('A', 'B', 'R')], b=120.0)
        for p, spend in (('A', 60.0), ('B', 25.0), ('R', 35.0)):
            eng.trade({p: eng.shares_for_spend(p, spend)})
        pr = eng.prices()
        self.assertAlmostEqual(pr['A'], GOLDEN['A'], delta=1e-7)
        self.assertAlmostEqual(pr['B'], GOLDEN['B'], delta=1e-7)
        self.assertAlmostEqual(pr['R'], GOLDEN['R'], delta=1e-7)
        self.assertEqual((round(pr['A'], 2), round(pr['B'], 2), round(pr['R'], 2)), (0.60, 0.46, 0.52))

    def test_rejects_contradictory_settlement(self):
        eng = ExactEngine(['A', 'B', 'R'], [('A', 'B', 'R')])
        eng.settle('A', True)
        eng.settle('B', True)
        with self.assertRaises(ValueError):
            eng.settle('R', True)


GOLDEN = dict(A=0.5990965027882311, B=0.4565903433480918, R=0.5213201819563652)


if __name__ == '__main__':
    unittest.main()
