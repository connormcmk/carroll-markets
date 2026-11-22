"""The pricing engine: the part of a Carroll market whose cost grows with
the size of the graph.

A Carroll market is a logarithmic market maker (LMSR) over the joint truth
assignments of its points, with every assignment that violates a NAND triple
removed. All trading is in point securities ("one share of A pays 1 if A is
true"), so the whole market state is one number per point: s[i], the net
shares of point i outstanding. The implied distribution over assignments x is

    P(x)  proportional to  exp( sum_i s[i] * x[i] / b )   for admissible x,

the cost function is C(s) = b * log Z(s), a trade costs C(after) - C(before),
and the price of a point is its marginal probability under P. In other words:
pricing is inference in a graphical model with one field per point and one
hard three-way constraint per NAND.

`Engine` is the interface every implementation must satisfy. `ExactEngine`
answers by listing every admissible assignment, which is exact and takes
time and memory proportional to 2^(number of points). It is the ground truth
that faster engines are measured against (see oracle.py, bench.py).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Nand = Tuple[str, str, str]          # (claim, counterpoint, relevance): not all three true


class Engine:
    """What the rest of the mechanism may ask of a pricing engine.

    Quantities are in shares and currency; `delta` maps point name -> shares
    bought (positive) or sold (negative) in one atomic trade.
    """

    points: List[str]
    nands: List[Nand]
    b: float

    def prices(self) -> Dict[str, float]:
        raise NotImplementedError

    def cost(self, delta: Mapping[str, float]) -> float:
        """Currency the trader pays for `delta` (negative = receives). No state change."""
        raise NotImplementedError

    def trade(self, delta: Mapping[str, float]) -> float:
        """Apply `delta`; return what the trader pays."""
        raise NotImplementedError

    def settle(self, point: str, value: bool) -> None:
        """Fix a point's truth value for good. Its price becomes 1 or 0."""
        raise NotImplementedError

    def outstanding(self) -> Dict[str, float]:
        raise NotImplementedError

    def add_point(self, name: str, offset: float = 0.0) -> None:
        raise NotImplementedError

    def add_nand(self, claim: str, counterpoint: str, relevance: str) -> None:
        raise NotImplementedError

    # ---- conveniences, defined once in terms of the above ----

    def joint_true(self, a: str, b: str) -> float:
        """The probability that points a and b are both true. Read from the
        cost function (the cost of an overwhelming purchase of both), so it
        changes nothing and works for any engine."""
        big = 40.0 * self.b
        return min(1.0, math.exp((self.cost({a: big, b: big}) - 2.0 * big) / self.b))

    def add_relevance(self, claim: str, counterpoint: str, name: str, entry_price: float = 0.01,
                      landed: bool = False) -> None:
        """Add a relevance point tying `counterpoint` to `claim`. Its price
        rises afterwards only when someone buys it. Nothing is done to any
        other point's weight.

        landed=False: `entry_price` is the price the point would have standing
        alone. Tied in, it lands a little lower. At the default 0.01 no other
        price moves by more than about a quarter of a cent.

        landed=True: `entry_price` is the price the point shows once tied in.
        The claim and the counterpoint then move by whatever a relevance at
        that price implies; nobody pays for that move."""
        if not 0.0 < entry_price < 1.0:
            raise ValueError('entry_price must be strictly between 0 and 1')
        odds = entry_price / (1.0 - entry_price)
        if landed:
            room = 1.0 - self.joint_true(claim, counterpoint)   # share of the market where the tie is not binding
            if room <= 1e-12:
                raise ValueError('claim and counterpoint are both certain; a relevance between them cannot be priced')
            odds /= room
        self.add_point(name, offset=self.b * math.log(odds))
        self.add_nand(claim, counterpoint, name)

    def price(self, point: str) -> float:
        return self.prices()[point]

    def shares_for_spend(self, point: str, spend: float, tol: float = 1e-10) -> float:
        """Shares of `point` that cost exactly `spend` (spend > 0)."""
        if spend <= 0:
            return 0.0
        lo, hi = 0.0, max(spend, 1.0)
        while self.cost({point: hi}) < spend:
            hi *= 2.0
            if hi > 1e12:
                raise ValueError(f'cannot spend {spend} on {point}: price is pinned')
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.cost({point: mid}) < spend:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tol * max(1.0, hi):
                break
        return 0.5 * (lo + hi)


class ExactEngine(Engine):
    """Ground truth by enumeration. Exponential in the number of points."""

    MAX_POINTS = 24

    def __init__(self, points: Sequence[str], nands: Iterable[Nand] = (), b: float = 100.0):
        self.points = list(points)
        if len(set(self.points)) != len(self.points):
            raise ValueError('duplicate point names')
        if len(self.points) > self.MAX_POINTS:
            raise ValueError(f'ExactEngine enumerates 2^n assignments; {len(self.points)} points is too many')
        self.b = float(b)
        self._idx = {p: i for i, p in enumerate(self.points)}
        self.nands = []
        for tri in nands:
            self._check_nand(tri)
            self.nands.append(tuple(tri))
        self._s = [0.0] * len(self.points)
        self._o = [0.0] * len(self.points)        # starting weights, in share units
        self._settled: Dict[int, int] = {}
        self._build()

    # ---- graph growth ----

    def add_point(self, name: str, offset: float = 0.0) -> None:
        if name in self._idx:
            raise ValueError(f'point {name!r} exists')
        if len(self.points) >= self.MAX_POINTS:
            raise ValueError('too many points for ExactEngine')
        self._idx[name] = len(self.points)
        self.points.append(name)
        self._s.append(0.0)
        self._o.append(float(offset))
        self._build()

    def add_nand(self, claim: str, counterpoint: str, relevance: str) -> None:
        tri = (claim, counterpoint, relevance)
        self._check_nand(tri)
        self.nands.append(tri)
        self._build()

    def _check_nand(self, tri) -> None:
        if len(tri) != 3 or len(set(tri)) != 3:
            raise ValueError(f'a NAND needs three distinct points, got {tri!r}')
        for p in tri:
            if p not in self._idx:
                raise ValueError(f'unknown point {p!r} in NAND {tri!r}')

    # ---- enumeration ----

    def _build(self) -> None:
        n = len(self.points)
        forbidden = [sum(1 << self._idx[p] for p in tri) for tri in self.nands]
        fixed_mask = sum(1 << i for i in self._settled)
        fixed_bits = sum(1 << i for i, v in self._settled.items() if v)
        self._outcomes = [w for w in range(1 << n)
                          if (w & fixed_mask) == fixed_bits
                          and not any((w & m) == m for m in forbidden)]
        if not self._outcomes:
            raise ValueError('no admissible assignment: the settled values contradict a NAND')
        self._members = [[k for k, w in enumerate(self._outcomes) if (w >> i) & 1] for i in range(n)]
        self._energy = [sum(self._s[i] + self._o[i] for i in range(n) if (w >> i) & 1) for w in self._outcomes]

    @property
    def n_outcomes(self) -> int:
        return len(self._outcomes)

    def _log_z(self, energy: List[float]) -> float:
        m = max(energy)
        return m / self.b + math.log(sum(math.exp((e - m) / self.b) for e in energy))

    def log_partition(self) -> float:
        return self._log_z(self._energy)

    def _shifted(self, delta: Mapping[str, float]) -> List[float]:
        e = list(self._energy)
        for p, d in delta.items():
            if d:
                for k in self._members[self._idx[p]]:
                    e[k] += d
        return e

    # ---- Engine ----

    def prices(self) -> Dict[str, float]:
        m = max(self._energy)
        w = [math.exp((e - m) / self.b) for e in self._energy]
        z = sum(w)
        return {p: sum(w[k] for k in self._members[i]) / z for i, p in enumerate(self.points)}

    def cost(self, delta: Mapping[str, float]) -> float:
        return self.b * (self._log_z(self._shifted(delta)) - self._log_z(self._energy))

    def trade(self, delta: Mapping[str, float]) -> float:
        new = self._shifted(delta)
        paid = self.b * (self._log_z(new) - self._log_z(self._energy))
        self._energy = new
        for p, d in delta.items():
            self._s[self._idx[p]] += d
        return paid

    def settle(self, point: str, value: bool) -> None:
        i = self._idx[point]
        if i in self._settled:
            raise ValueError(f'{point!r} is already settled')
        self._settled[i] = 1 if value else 0
        self._build()

    def outstanding(self) -> Dict[str, float]:
        return {p: self._s[i] for i, p in enumerate(self.points)}

    def admissible(self) -> List[Dict[str, int]]:
        """Every truth assignment the NANDs and settlements allow."""
        return [{p: (w >> i) & 1 for i, p in enumerate(self.points)} for w in self._outcomes]

    def settled(self) -> Dict[str, bool]:
        return {self.points[i]: bool(v) for i, v in self._settled.items()}

    # ---- an independent second computation, for the tests ----

    def reference_prices_and_logz(self) -> Tuple[Dict[str, float], float]:
        """Recompute from the share vector alone, sharing no cached state with
        the methods above: no stored assignment list, no stored energies."""
        n = len(self.points)
        z = 0.0
        num = [0.0] * n
        f = [a + c for a, c in zip(self._s, self._o)]
        shift = max(0.0, sum(max(v, 0.0) for v in f))
        for w in range(1 << n):
            x = [(w >> i) & 1 for i in range(n)]
            if any(all(x[self._idx[p]] for p in tri) for tri in self.nands):
                continue
            if any(x[i] != v for i, v in self._settled.items()):
                continue
            wt = math.exp((sum(f[i] for i in range(n) if x[i]) - shift) / self.b)
            z += wt
            for i in range(n):
                if x[i]:
                    num[i] += wt
        return {p: num[i] / z for i, p in enumerate(self.points)}, shift / self.b + math.log(z)
