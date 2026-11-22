"""Ground-truth comparison for pricing engines.

A candidate engine is run through the same trade sequence as ExactEngine on
the same graph, and scored on how far its prices and trade costs drift from
the exact ones, and on four things that decide whether an error can be exploited: a round
trip costs nothing (buy then sell the same shares); the order of trades does
not change the total paid; no trade locks in a profit whatever the world turns
out to be (`guaranteed_profit`); and how much a trader who knows the exact
prices can expect to take from one trade (`informed_profit`).

Graph families, by what makes them hard:
    star      one claim, k counterpoints           (tree-like: triangles share one point)
    tower     mitigations stacked on a relevance   (tree-like, deep)
    chain     each point counters the previous     (tree-like, long)
    shared    one counterpoint against k claims    (tree-like, wide hub)
    ring      k claims, each countering the next   (one long loop)
    dense     random NANDs over few points         (many short loops)
    board     policies with shared counterpoints   (long loops; a few first-level mitigations)
    argument  grown move by move, any depth, any   (mitigations of mitigations, points in
              point reusable in any role            several roles, loops of every length)
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

from .engine import Engine, ExactEngine, Nand

Graph = Tuple[List[str], List[Nand]]


def star(k: int) -> Graph:
    pts, nands = ['A'], []
    for j in range(k):
        pts += [f'B{j}', f'R{j}']
        nands.append(('A', f'B{j}', f'R{j}'))
    return pts, nands


def tower(depth: int) -> Graph:
    pts, nands = ['A', 'B', 'R0'], [('A', 'B', 'R0')]
    for d in range(depth):
        pts += [f'M{d}', f'R{d + 1}']
        nands.append((f'R{d}', f'M{d}', f'R{d + 1}'))
    return pts, nands


def chain(k: int) -> Graph:
    pts, nands = ['P0'], []
    for j in range(k):
        pts += [f'P{j + 1}', f'R{j}']
        nands.append((f'P{j}', f'P{j + 1}', f'R{j}'))
    return pts, nands


def shared(k: int) -> Graph:
    pts, nands = ['E'], []
    for j in range(k):
        pts += [f'A{j}', f'R{j}']
        nands.append((f'A{j}', 'E', f'R{j}'))
    return pts, nands


def ring(k: int) -> Graph:
    pts = [f'A{j}' for j in range(k)] + [f'R{j}' for j in range(k)]
    nands = [(f'A{j}', f'A{(j + 1) % k}', f'R{j}') for j in range(k)]
    return pts, nands


def dense(n: int, n_nands: int, seed: int = 0) -> Graph:
    rng = random.Random(seed)
    pts = [f'X{j}' for j in range(n)]
    nands = set()
    while len(nands) < n_nands:
        nands.add(tuple(rng.sample(pts, 3)))
    return pts, sorted(nands)


def board(n_policies: int, n_counterpoints: int, attach: int, n_mitigations: int = 0, seed: int = 0) -> Graph:
    """Shaped like a governance board: policies, and counterpoints that each bear
    on `attach` policies chosen at random (one relevance point per attachment),
    plus mitigations aimed at randomly chosen relevance points. Counterpoints
    shared between policies are what close loops."""
    rng = random.Random(seed)
    pts = [f'P{i}' for i in range(n_policies)]
    nands, rels = [], []
    for j in range(n_counterpoints):
        pts.append(f'C{j}')
        for i in rng.sample(range(n_policies), min(attach, n_policies)):
            r = f'R{j}_{i}'
            pts.append(r)
            rels.append(r)
            nands.append((f'P{i}', f'C{j}', r))
    for k in range(n_mitigations):
        r = rng.choice(rels)
        pts += [f'M{k}', f'RM{k}']
        nands.append((r, f'M{k}', f'RM{k}'))
    return pts, nands


def argument(n_policies: int, n_moves: int, reuse: float = 0.3, seed: int = 0,
             aim: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> Graph:
    """Grown the way an argument grows, with no limit on depth or on the roles a
    point plays. Each move aims a point at a target through a new relevance
    point. The target is a policy, an earlier counterpoint (a counterpoint to a
    counterpoint) or an earlier relevance point (a mitigation, and mitigations
    of mitigations to any depth), chosen with weights `aim`. With probability
    `reuse` the point aimed is one already on the board, so one point can be a
    counterpoint in one place and a mitigation in another, policies can counter
    each other, and a relevance point can itself be used as a counterpoint;
    otherwise the point is new."""
    rng = random.Random(seed)
    policies = [f'P{i}' for i in range(n_policies)]
    pts, others, rels, nands, used = list(policies), [], [], [], set()
    while len(nands) < n_moves:
        pools = [policies, others, rels]
        target = rng.choice(rng.choices(pools, [w if pool else 0.0 for w, pool in zip(aim, pools)])[0])
        if others and rng.random() < reuse:
            c = rng.choice(rels) if rng.random() < 0.15 else rng.choice(others + policies)
            if c == target or (target, c) in used:
                continue
        else:
            c = f'C{len(others)}'
            others.append(c)
            pts.append(c)
        r = f'R{len(rels)}'
        rels.append(r)
        pts.append(r)
        used.add((target, c))
        nands.append((target, c, r))
    return pts, nands


FAMILIES: Dict[str, Callable[..., Graph]] = dict(star=star, tower=tower, chain=chain, shared=shared,
                                                 ring=ring, dense=dense, board=board, argument=argument)


def guaranteed_profit(cand: Engine, exact: ExactEngine, tries: int = 300, seed: int = 0, size: float = 150.0) -> float:
    """The most a trader can lock in against `cand` whatever the world turns out
    to be: the best found value of (worst-case payout of a trade) - (its cost).
    Positive means free money. A sound market maker never offers it."""
    rng = random.Random(seed)
    worlds = exact.admissible()
    pts = exact.points

    def locked(d):
        return min(sum(d[p] * x[p] for p in d) for x in worlds) - cand.cost(d)

    best = 0.0
    for _ in range(tries):
        chosen = rng.sample(pts, min(len(pts), rng.choice((1, 2, 3))))
        d = {p: rng.uniform(-size, size) for p in chosen}
        v = locked(d)
        for _ in range(3):                      # shrink toward the best scale of this direction
            for f in (0.5, 0.25, 0.1):
                d2 = {p: q * f for p, q in d.items()}
                v2 = locked(d2)
                if v2 > v:
                    d, v = d2, v2
        best = max(best, v)
    return best


def informed_profit(cand: Engine, exact: ExactEngine, size: float = 200.0) -> float:
    """The most a trader who knows the exact prices can expect to make from one
    single-point trade against `cand`: expected payout at exact prices minus
    cost. Zero when cand's prices are exact; it measures what a pricing error
    is worth to someone who can see it."""
    truth = exact.prices()
    best = 0.0
    for p in exact.points:
        lo, hi = -size, size
        f = lambda d: d * truth[p] - cand.cost({p: d})
        for _ in range(60):                     # the gain is concave in d when cost is convex
            a, b2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if f(a) < f(b2):
                lo = a
            else:
                hi = b2
        best = max(best, f(0.5 * (lo + hi)))
    return best


def trade_sequence(points: Sequence[str], n: int, seed: int = 0, size: float = 40.0) -> List[Dict[str, float]]:
    """Random single-point trades that never take a point's outstanding shares below zero."""
    rng = random.Random(seed)
    out_s = {p: 0.0 for p in points}
    seq = []
    for _ in range(n):
        p = rng.choice(list(points))
        d = rng.uniform(0.2, 1.0) * size
        if out_s[p] > 0 and rng.random() < 0.35:
            d = -min(d, out_s[p])
        out_s[p] += d
        seq.append({p: d})
    return seq


def compare(make_candidate: Callable[[List[str], List[Nand], float], Engine], graph: Graph,
            n_trades: int = 60, seed: int = 0, b: float = 100.0, profit_tries: int = 300) -> Dict[str, float]:
    """Score a candidate engine against the exact one on one graph."""
    pts, nands = graph
    exact = ExactEngine(pts, nands, b=b)
    cand = make_candidate(list(pts), list(nands), b)
    seq = trade_sequence(pts, n_trades, seed=seed)
    price_err, cost_err, cost_scale, round_trip = 0.0, 0.0, 0.0, 0.0
    paid_exact = paid_cand = 0.0
    for delta in seq:
        (p, d), = delta.items()
        back = cand.cost({p: d})
        ce, cc = exact.trade(delta), cand.trade(delta)
        round_trip = max(round_trip, abs(back + cand.cost({p: -d})))
        paid_exact += ce
        paid_cand += cc
        cost_err = max(cost_err, abs(cc - ce))
        cost_scale = max(cost_scale, abs(ce))
        pe, pc = exact.prices(), cand.prices()
        price_err = max(price_err, max(abs(pe[q] - pc[q]) for q in pts))
    # order independence: the same trades, reversed, from a fresh engine
    again = make_candidate(list(pts), list(nands), b)
    paid_reversed = sum(again.trade(delta) for delta in reversed(seq))
    return dict(points=len(pts), nands=len(nands),
                guaranteed_profit=guaranteed_profit(cand, exact, tries=profit_tries, seed=seed),
                informed_profit=informed_profit(cand, exact),
                max_price_error=price_err,
                max_cost_error=cost_err,
                max_cost_error_relative=cost_err / cost_scale if cost_scale else 0.0,
                total_paid_error=abs(paid_cand - paid_exact),
                round_trip_leak=round_trip,
                order_dependence=abs(paid_cand - paid_reversed))


def default_suite(max_points: int = 15) -> List[Tuple[str, Graph]]:
    """The fixed set of small graphs a candidate is scored on."""
    out = []
    for k in (1, 3, 5, 7):
        out.append((f'star({k})', star(k)))
    for d in (1, 3, 5):
        out.append((f'tower({d})', tower(d)))
    for k in (2, 4, 6):
        out.append((f'chain({k})', chain(k)))
    for k in (2, 4, 6):
        out.append((f'shared({k})', shared(k)))
    for k in (3, 5, 7):
        out.append((f'ring({k})', ring(k)))
    for n, m, s in ((6, 5, 1), (8, 9, 2), (10, 14, 3), (12, 20, 4)):
        out.append((f'dense({n},{m})', dense(n, m, s)))
    return [(name, g) for name, g in out if len(g[0]) <= max_points]


def score(make_candidate, max_points: int = 15, n_trades: int = 60) -> List[Dict]:
    rows = []
    for name, g in default_suite(max_points):
        r = compare(make_candidate, g, n_trades=n_trades, seed=len(g[0]))
        r['graph'] = name
        rows.append(r)
    return rows
