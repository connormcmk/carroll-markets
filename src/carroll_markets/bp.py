"""A pricing engine with no partitioning: loopy belief propagation.

Every point repeatedly tells each NAND it belongs to how likely it is to be
true given what its other NANDs say, and every NAND tells each of its points
how much room the other two leave it. On a graph with no loops this settles
on the exact prices. With loops it settles (usually) on an approximation.
The cost function is the matching approximation to b log Z, the Bethe free
energy, so the cost of a trade is still the change in one function of the
share vector. Work per sweep is proportional to the number of NANDs,
whatever the graph's shape.

Messages. A NAND over (a, b, r) allows everything except all three true, so
what it tells a is the ratio  c = 1 - mu_b * mu_r  (weight of a true over
weight of a false), where mu is what the other two points report about
themselves. A point's report to one NAND is its own weight times the ratios
from its other NANDs, turned into a probability.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

from .engine import Engine, Nand

TINY = 1e-300


def _log1pexp(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class BeliefPropagationEngine(Engine):
    def __init__(self, points: Sequence[str], nands: Iterable[Nand] = (), b: float = 100.0,
                 damping: float = 0.3, tol: float = 1e-11, max_sweeps: int = 2000):
        self.points = list(points)
        self.b = float(b)
        self.damping, self.tol, self.max_sweeps = damping, tol, max_sweeps
        self._idx = {p: i for i, p in enumerate(self.points)}
        self.nands = [tuple(t) for t in nands]
        self._s = [0.0] * len(self.points)
        self._f = [0.0] * len(self.points)
        self._settled: Dict[int, int] = {}
        self.sweeps_last = 0
        self._wire()
        self._logz = self._run(self._f, commit=True)

    def _wire(self) -> None:
        self._tri = [tuple(self._idx[p] for p in t) for t in self.nands]
        self._slots: List[List[tuple]] = [[] for _ in self.points]     # point -> [(nand, position)]
        for k, tri in enumerate(self._tri):
            for pos, v in enumerate(tri):
                self._slots[v].append((k, pos))
        self._logc = [[0.0, 0.0, 0.0] for _ in self._tri]              # NAND -> point, log ratio
        self._mu = [[0.5, 0.5, 0.5] for _ in self._tri]                # point -> NAND, probability true

    def add_point(self, name: str, offset: float = 0.0) -> None:
        if name in self._idx:
            raise ValueError(f'point {name!r} exists')
        self._idx[name] = len(self.points)
        self.points.append(name)
        self._s.append(0.0)
        self._f.append(float(offset))
        self._slots.append([])
        self._logz = self._run(self._f, commit=True)

    def add_nand(self, claim: str, counterpoint: str, relevance: str) -> None:
        self.nands.append((claim, counterpoint, relevance))
        self._wire()
        self._logz = self._run(self._f, commit=True)

    def _theta(self, v: int, f) -> float:
        if v in self._settled:
            return 700.0 if self._settled[v] else -700.0
        return f[v] / self.b

    def _run(self, f, commit: bool) -> float:
        """Iterate from the stored messages to a fixed point for fields `f`;
        return the Bethe estimate of log Z."""
        logc = [list(r) for r in self._logc]
        mu = [list(r) for r in self._mu]
        theta = [self._theta(v, f) for v in range(len(self.points))]
        d = self.damping
        sweeps = 0
        for sweeps in range(1, self.max_sweeps + 1):
            change = 0.0
            for v, slots in enumerate(self._slots):
                total = theta[v] + sum(logc[k][pos] for k, pos in slots)
                for k, pos in slots:
                    new = _sigmoid(total - logc[k][pos])
                    new = d * mu[k][pos] + (1 - d) * new
                    change = max(change, abs(new - mu[k][pos]))
                    mu[k][pos] = new
            for k in range(len(self._tri)):
                m = mu[k]
                for pos in range(3):
                    o1, o2 = (pos + 1) % 3, (pos + 2) % 3
                    new = math.log(max(1.0 - m[o1] * m[o2], TINY))
                    change = max(change, abs(new - logc[k][pos]))
                    logc[k][pos] = new
            if change < self.tol:
                break
        logz = 0.0
        for v, slots in enumerate(self._slots):
            logz += _log1pexp(theta[v] + sum(logc[k][pos] for k, pos in slots))
        for k in range(len(self._tri)):
            m = mu[k]
            logz += math.log(max(1.0 - m[0] * m[1] * m[2], TINY))
            for pos in range(3):
                logz -= math.log(max((1.0 - m[pos]) + m[pos] * math.exp(logc[k][pos]), TINY))
        if commit:
            self._logc, self._mu, self.sweeps_last = logc, mu, sweeps
        return logz

    def prices(self) -> Dict[str, float]:
        out = {}
        for v, slots in enumerate(self._slots):
            out[self.points[v]] = _sigmoid(self._theta(v, self._f) + sum(self._logc[k][pos] for k, pos in slots))
        return out

    def cost(self, delta: Mapping[str, float]) -> float:
        f = list(self._f)
        for p, d in delta.items():
            f[self._idx[p]] += d
        return self.b * (self._run(f, commit=False) - self._logz)

    def trade(self, delta: Mapping[str, float]) -> float:
        for p, d in delta.items():
            self._s[self._idx[p]] += d
            self._f[self._idx[p]] += d
        new = self._run(self._f, commit=True)
        paid = self.b * (new - self._logz)
        self._logz = new
        return paid

    def settle(self, point: str, value: bool) -> None:
        v = self._idx[point]
        if v in self._settled:
            raise ValueError(f'{point!r} is already settled')
        self._settled[v] = 1 if value else 0
        self._logz = self._run(self._f, commit=True)

    def outstanding(self) -> Dict[str, float]:
        return {p: self._s[i] for i, p in enumerate(self.points)}
