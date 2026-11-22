"""An exact pricing engine whose cost depends on how tangled the graph is, not
on how many points it has.

It computes the same quantities as ExactEngine (log Z and every point's
probability) by eliminating points one at a time and passing small tables
between neighbouring groups of points, the junction-tree method. Loops are
handled exactly. The cost of a price read is the total size of those tables:
each group of k points needs a table of 2^k numbers, so what matters is the
largest group, reported as `width` (largest group size minus one). A claim
with a hundred counterpoints has width 2. A graph where many points are all
tied to one another has large width, and this engine refuses it
(`MAX_WIDTH`) rather than run forever.

Layout. Points are eliminated in a greedy order (fewest remaining
neighbours first). Eliminating point v forms the group [v] + its remaining
neighbours; v is always position 0, so a table index's lowest bit is v's
value and the rest index the separator shared with the parent group (the
group of whichever neighbour is eliminated next). Each group owns one
point's weight [1, exp(s/b)] and the NANDs whose earliest-eliminated point
is v.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .engine import Engine, Nand


class TooTangled(ValueError):
    pass


def eliminate(n: int, tris, cap: Optional[int] = None):
    """Greedy elimination order: fewest remaining neighbours first, ties by
    fewest edges to add. Returns (groups, when): group k is [v] + v's remaining
    neighbours when v is eliminated k-th; when[v] = k."""
    adj = [set() for _ in range(n)]
    for tri in tris:
        for u in tri:
            adj[u].update(v for v in tri if v != u)
    when = [0] * n
    groups: List[List[int]] = []
    buckets: Dict[int, set] = {}
    for v in range(n):
        buckets.setdefault(len(adj[v]), set()).add(v)

    def fill(v):
        nb = list(adj[v])
        return sum(1 for i, x in enumerate(nb) for y in nb[i + 1:] if y not in adj[x])

    for k in range(n):
        d = min(deg for deg, vs in buckets.items() if vs)
        cands = buckets[d]
        v = min(cands, key=lambda u: (fill(u), u)) if d > 1 and len(cands) > 1 else min(cands)
        nb = sorted(adj[v])
        if cap is not None and len(nb) > cap:
            raise TooTangled(f'{len(nb) + 1} points are mutually tied; tables of 2^{len(nb) + 1} entries are too large')
        for u in nb + [v]:
            buckets[len(adj[u])].discard(u)
        for i, x in enumerate(nb):
            adj[x].discard(v)
            for y in nb[i + 1:]:
                adj[x].add(y)
                adj[y].add(x)
        for u in nb:
            buckets.setdefault(len(adj[u]), set()).add(u)
        when[v] = k
        groups.append([v] + nb)
    return groups, when


def width_of(points: Sequence[str], nands: Iterable[Nand]) -> int:
    """Largest group size minus one under the greedy order, with no size limit."""
    idx = {p: i for i, p in enumerate(points)}
    groups, _ = eliminate(len(points), [tuple(idx[p] for p in tri) for tri in nands])
    return max((len(g) for g in groups), default=1) - 1


class JunctionTreeEngine(Engine):
    MAX_WIDTH = 18

    def __init__(self, points: Sequence[str], nands: Iterable[Nand] = (), b: float = 100.0):
        self.points = list(points)
        if len(set(self.points)) != len(self.points):
            raise ValueError('duplicate point names')
        self.b = float(b)
        self._idx = {p: i for i, p in enumerate(self.points)}
        self.nands: List[Nand] = []
        for tri in nands:
            self._check_nand(tri)
            self.nands.append(tuple(tri))
        self._s = [0.0] * len(self.points)        # net shares outstanding
        self._f = [0.0] * len(self.points)        # shares plus the point's starting weight
        self._settled: Dict[int, int] = {}
        self._build()

    # ---- graph growth ----

    def add_point(self, name: str, offset: float = 0.0) -> None:
        if name in self._idx:
            raise ValueError(f'point {name!r} exists')
        self._idx[name] = len(self.points)
        self.points.append(name)
        self._s.append(0.0)
        self._f.append(float(offset))
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

    # ---- structure ----

    def _build(self) -> None:
        n = len(self.points)
        tris = [tuple(self._idx[p] for p in tri) for tri in self.nands]
        self._group, when = eliminate(n, tris, cap=self.MAX_WIDTH)
        self._home = when                    # point -> index of the group that owns it
        self._parent: List[Optional[int]] = []
        self._children: List[List[int]] = [[] for _ in range(n)]
        self._sepmap: List[Optional[List[int]]] = []     # parent table index -> this group's separator index
        for k, g in enumerate(self._group):
            if len(g) == 1:
                self._parent.append(None)
                self._sepmap.append(None)
                continue
            par = min(when[u] for u in g[1:])
            self._parent.append(par)
            self._children[par].append(k)
            pos = {u: t for t, u in enumerate(self._group[par])}
            bits = [pos[u] for u in g[1:]]
            self._sepmap.append([sum(((J >> bt) & 1) << t for t, bt in enumerate(bits))
                                 for J in range(1 << len(self._group[par]))])
        # each NAND lives in the group of its earliest-eliminated point
        self._allowed: List[List[float]] = [[1.0] * (1 << len(g)) for g in self._group]
        for tri in tris:
            k = min(when[u] for u in tri)
            pos = {u: t for t, u in enumerate(self._group[k])}
            mask = sum(1 << pos[u] for u in tri)
            tbl = self._allowed[k]
            for j in range(len(tbl)):
                if (j & mask) == mask:
                    tbl[j] = 0.0
        self.width = max((len(g) for g in self._group), default=1) - 1
        self.table_entries = sum(1 << len(g) for g in self._group)
        self._pot: List[List[float]] = [[] for _ in range(n)]
        self._up: List[List[float]] = [[] for _ in range(n)]      # message to parent, normalised
        self._scale = [0.0] * n                                   # log of what was divided out
        self._prices: Optional[Dict[str, float]] = None
        self._logz = self._upward(range(n), self._f, self._settled, commit=True)
        if self._logz == -math.inf:
            raise ValueError('no admissible assignment')

    # ---- the upward pass: log Z ----

    def _weight(self, v: int, s, settled):
        """A point's two weights, scaled to avoid overflow, and the log of the scale."""
        if v in settled:
            return ((0.0, 1.0) if settled[v] else (1.0, 0.0)), (s[v] / self.b if settled[v] else 0.0)
        x = s[v] / self.b
        m = max(0.0, x)
        return (math.exp(-m), math.exp(x - m)), m

    def _upward(self, groups, s, settled, commit: bool) -> float:
        """Recompute the listed groups (which must include every ancestor of any
        group listed, in elimination order) and return log Z. With commit=False
        nothing stored is changed."""
        pot, up, scale = ({}, {}, {}) if not commit else (None, None, None)

        def get_up(c):
            return up[c] if (up is not None and c in up) else self._up[c]

        for k in groups:
            g = self._group[k]
            (w0, w1), m = self._weight(g[0], s, settled)
            allowed = self._allowed[k]
            t = [allowed[j] * (w1 if j & 1 else w0) for j in range(len(allowed))]
            for c in self._children[k]:
                msg, sm = get_up(c), self._sepmap[c]
                t = [t[J] * msg[sm[J]] for J in range(len(t))]
            half = len(t) >> 1
            msg = [t[2 * i] + t[2 * i + 1] for i in range(half)]
            total = sum(msg)
            if total > 0:
                msg = [x / total for x in msg]
                sc = m + math.log(total)
            else:
                sc = -math.inf
            if commit:
                self._pot[k], self._up[k], self._scale[k] = t, msg, sc
            else:
                pot[k], up[k], scale[k] = t, msg, sc
        if commit:
            return sum(self._scale)
        return sum(scale.get(k, self._scale[k]) for k in range(len(self._group)))

    def _ancestry(self, vs) -> List[int]:
        """Groups owning the given points, plus all their ancestors, in elimination order."""
        seen = set()
        for v in vs:
            k = self._home[v]
            while k is not None and k not in seen:
                seen.add(k)
                k = self._parent[k]
        return sorted(seen)

    # ---- the downward pass: every price ----

    def prices(self) -> Dict[str, float]:
        if self._prices is not None:
            return dict(self._prices)
        n = len(self._group)
        down: List[Optional[List[float]]] = [None] * n
        out = {}
        for k in range(n - 1, -1, -1):
            t = self._pot[k]
            if down[k] is not None:
                d = down[k]
                t = [t[j] * d[j >> 1] for j in range(len(t))]
            total = sum(t)
            out[self.points[self._group[k][0]]] = sum(t[j] for j in range(1, len(t), 2)) / total
            for c in self._children[k]:
                sm, upc = self._sepmap[c], self._up[c]
                acc = [0.0] * len(upc)
                for J in range(len(t)):
                    acc[sm[J]] += t[J]
                down[c] = [a / u if u > 0 else 0.0 for a, u in zip(acc, upc)]
        self._prices = out
        return dict(out)

    # ---- Engine ----

    def cost(self, delta: Mapping[str, float]) -> float:
        moved = [self._idx[p] for p, d in delta.items() if d]
        if not moved:
            return 0.0
        f = list(self._f)
        for p, d in delta.items():
            f[self._idx[p]] += d
        return self.b * (self._upward(self._ancestry(moved), f, self._settled, commit=False) - self._logz)

    def trade(self, delta: Mapping[str, float]) -> float:
        moved = [self._idx[p] for p, d in delta.items() if d]
        if not moved:
            return 0.0
        for p, d in delta.items():
            self._s[self._idx[p]] += d
            self._f[self._idx[p]] += d
        new = self._upward(self._ancestry(moved), self._f, self._settled, commit=True)
        paid = self.b * (new - self._logz)
        self._logz = new
        self._prices = None
        return paid

    def settle(self, point: str, value: bool) -> None:
        v = self._idx[point]
        if v in self._settled:
            raise ValueError(f'{point!r} is already settled')
        trial = dict(self._settled)
        trial[v] = 1 if value else 0
        if self._upward(self._ancestry([v]), self._f, trial, commit=False) == -math.inf:
            raise ValueError('no admissible assignment: the settled values contradict a NAND')
        self._settled = trial
        self._logz = self._upward(self._ancestry([v]), self._f, self._settled, commit=True)
        self._prices = None

    def outstanding(self) -> Dict[str, float]:
        return {p: self._s[i] for i, p in enumerate(self.points)}

    def log_partition(self) -> float:
        return self._logz
