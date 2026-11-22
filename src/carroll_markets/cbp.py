"""Belief propagation between small exact clusters.

Plain belief propagation (bp.py) goes wrong where two NANDs share two
points: the loop between them is as short as a loop can be, and each NAND
hears its own message back at once. Here any two NANDs that share two points
are merged into one cluster, and clusters that share two points are merged
again, until every pair of clusters shares at most one point (or a cluster
would pass `max_cluster` points). Each cluster keeps a table of which
assignments of its own points are admissible and is summed exactly; messages
then pass between clusters and points as before. With no merging this is
bp.py exactly. If everything merges into one cluster it is exact enumeration.

The cost function is the matching Bethe approximation to b log Z, so a trade
still costs the change in one function of the share vector.
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


def clusters_of(tris: Sequence[tuple], max_cluster: int = 14) -> List[List[int]]:
    """Group NANDs (given as index triples) so that no two groups share two
    points, as far as `max_cluster` allows. Returns lists of NAND indices."""
    groups = [[k] for k in range(len(tris))]
    scope = [set(t) for t in tris]
    merged = True
    while merged:
        merged = False
        where: Dict[int, List[int]] = {}
        for g, sc in enumerate(scope):
            if sc is not None:
                for v in sc:
                    where.setdefault(v, []).append(g)
        for g in range(len(scope)):
            if scope[g] is None:
                continue
            seen: Dict[int, int] = {}
            for v in scope[g]:
                for h in where.get(v, ()):
                    if h != g and scope[h] is not None:
                        seen[h] = seen.get(h, 0) + 1
            for h, shared in seen.items():
                if shared >= 2 and scope[h] is not None and len(scope[g] | scope[h]) <= max_cluster:
                    scope[g] |= scope[h]
                    groups[g] += groups[h]
                    scope[h], groups[h] = None, []
                    merged = True
            if merged:
                break
    return [g for g in groups if g]


class ClusterBPEngine(Engine):
    def __init__(self, points: Sequence[str], nands: Iterable[Nand] = (), b: float = 100.0,
                 max_cluster: int = 14, damping: float = 0.3, tol: float = 1e-11, max_sweeps: int = 2000):
        self.points = list(points)
        self.b = float(b)
        self.max_cluster, self.damping, self.tol, self.max_sweeps = max_cluster, damping, tol, max_sweeps
        self._idx = {p: i for i, p in enumerate(self.points)}
        self.nands = [tuple(t) for t in nands]
        self._s = [0.0] * len(self.points)
        self._f = [0.0] * len(self.points)
        self._settled: Dict[int, int] = {}
        self.sweeps_last = 0
        self._wire()
        self._logz = self._run(self._f, commit=True)

    # ---- structure ----

    def _wire(self) -> None:
        tris = [tuple(self._idx[p] for p in t) for t in self.nands]
        self._scope: List[List[int]] = []          # cluster -> its points
        self._ok: List[List[int]] = []             # cluster -> admissible assignments (bitmasks over its points)
        for group in clusters_of(tris, self.max_cluster):
            pts = sorted({v for k in group for v in tris[k]})
            pos = {v: t for t, v in enumerate(pts)}
            masks = [sum(1 << pos[v] for v in tris[k]) for k in group]
            self._scope.append(pts)
            self._ok.append([x for x in range(1 << len(pts)) if all((x & m) != m for m in masks)])
        self._slots: List[List[tuple]] = [[] for _ in self.points]     # point -> [(cluster, position)]
        for c, pts in enumerate(self._scope):
            for t, v in enumerate(pts):
                self._slots[v].append((c, t))
        self._logc = [[0.0] * len(pts) for pts in self._scope]         # cluster -> point, log ratio
        self._mu = [[0.5] * len(pts) for pts in self._scope]           # point -> cluster, probability true
        self.largest_cluster = max((len(p) for p in self._scope), default=0)
        self.table_entries = sum(len(ok) for ok in self._ok)

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

    # ---- the iteration ----

    def _theta(self, v: int, f) -> float:
        if v in self._settled:
            return 700.0 if self._settled[v] else -700.0
        return f[v] / self.b

    @staticmethod
    def _sums(ok, m):
        """For one cluster with incoming probabilities m: the total weight of
        its admissible assignments, and for each of its points the weight with
        that point true, each computed with that point's own message left out."""
        k = len(m)
        if k == 3 and len(ok) == 7:                       # a lone NAND, in closed form
            a, b, r = m
            return 1.0 - a * b * r, None
        w1 = m
        w0 = [1.0 - x for x in m]
        total = 0.0
        on = [0.0] * k                                    # weight with point t true, t's factor removed
        off = [0.0] * k
        for x in ok:
            pre = [1.0] * (k + 1)                         # products of the others, from each side
            for t in range(k):
                pre[t + 1] = pre[t] * (w1[t] if (x >> t) & 1 else w0[t])
            total += pre[k]
            suf = 1.0
            for t in range(k - 1, -1, -1):
                rest = pre[t] * suf
                if (x >> t) & 1:
                    on[t] += rest
                    suf *= w1[t]
                else:
                    off[t] += rest
                    suf *= w0[t]
        return total, (on, off)

    def _run(self, f, commit: bool) -> float:
        logc = [list(r) for r in self._logc]
        mu = [list(r) for r in self._mu]
        theta = [self._theta(v, f) for v in range(len(self.points))]
        d = self.damping
        sweeps = 0
        for sweeps in range(1, self.max_sweeps + 1):
            change = 0.0
            for v, slots in enumerate(self._slots):
                total = theta[v] + sum(logc[c][t] for c, t in slots)
                for c, t in slots:
                    new = _sigmoid(total - logc[c][t])
                    new = d * mu[c][t] + (1 - d) * new
                    change = max(change, abs(new - mu[c][t]))
                    mu[c][t] = new
            for c, ok in enumerate(self._ok):
                m = mu[c]
                _, parts = self._sums(ok, m)
                for t in range(len(m)):
                    if parts is None:
                        o1, o2 = (t + 1) % 3, (t + 2) % 3
                        new = math.log(max(1.0 - m[o1] * m[o2], TINY))
                    else:
                        new = math.log(max(parts[0][t], TINY)) - math.log(max(parts[1][t], TINY))
                    change = max(change, abs(new - logc[c][t]))
                    logc[c][t] = new
            if change < self.tol:
                break
        logz = 0.0
        for v, slots in enumerate(self._slots):
            logz += _log1pexp(theta[v] + sum(logc[c][t] for c, t in slots))
        for c, ok in enumerate(self._ok):
            m = mu[c]
            total, _ = self._sums(ok, m)
            logz += math.log(max(total, TINY))
            for t in range(len(m)):
                logz -= math.log(max((1.0 - m[t]) + m[t] * math.exp(logc[c][t]), TINY))
        if commit:
            self._logc, self._mu, self.sweeps_last = logc, mu, sweeps
        return logz

    # ---- Engine ----

    def prices(self) -> Dict[str, float]:
        return {self.points[v]: _sigmoid(self._theta(v, self._f) + sum(self._logc[c][t] for c, t in slots))
                for v, slots in enumerate(self._slots)}

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
