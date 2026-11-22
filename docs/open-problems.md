# Open problems

## 1. Tractable price functions

A price read is a marginal of a distribution over up to `2^n` assignments. A useful
market needs hundreds of points and a price read in milliseconds. What exists:

**Exact, by partitioning.** `JunctionTreeEngine` is exact on any graph, loops included,
and its cost depends on the treewidth, not the number of points. A claim with a hundred
counterpoints is cheap. What makes a graph expensive is counterpoints that each bear on
several claims: the groups of mutually tied points grow, and each group of `k` points
needs a table of `2^k` entries. The engine refuses graphs whose groups pass 19 points
(`MAX_WIDTH = 18`). `python3 -m carroll_markets.survey` measures how wide randomly
generated boards get as counterpoints are shared more.

Open: better elimination orders; conditioning on a few hub points to cut wide graphs into
narrow ones; bounded-width approximations that give an upper bound on the cost function.

**Local, by message passing.** `BeliefPropagationEngine` does no partitioning. Its cost per
sweep is linear in the number of NANDs whatever the graph's shape, and it is exact when
the graph has no loops. What the tests record on graphs with loops:

- A round trip is still free, the order of trades still does not matter, and a search for
  a trade that locks in a profit finds none (`tests/test_bp.py`, `tests/test_cbp.py`).
  These hold because the cost of a trade is still the change in one function of the state.
- On a 110-point board where each counterpoint bears on three claims, every price is
  within 0.005 of exact (`tests/test_bp.py`).
- It goes wrong where two NANDs share two points, for example when a point counters a
  claim and also counters its own relevance there. On one such graph the worst price is
  off by more than 0.01 (`tests/test_cbp.py`).
- `ClusterBPEngine` merges NANDs that share two points and sums each cluster exactly. On
  that same graph its error is below one part in a million, and on the argument-shaped
  test graphs (nested arguments, points reused in several roles) every price is within
  0.005 of exact (`tests/test_cbp.py`).

Open, and deciding whether a local engine can ever be used:

1. *Adversarial states.* The figures above are after random trades. A trader searching
   for the market state where the approximation is worst may find larger errors and trade
   against them. `oracle.informed_profit` measures what an error is worth to a trader who
   knows the exact prices; it has not yet been maximised over states.
2. *Bias.* The small errors that remain are not symmetric. In our runs they lean in a
   consistent direction for a given kind of point. A consistent lean is what a trader can
   farm, so it needs to be measured and either removed or bounded.
3. *Convergence at extreme prices.* Whether the iteration always settles, and on a single
   answer, when many prices are near 0 or 1. Two resting states would break path
   independence.
4. *Bounded loss.* The exact cost function bounds the market maker's loss. The Bethe
   approximation carries no such guarantee. Tree-reweighted approximations give an upper
   bound on the log partition function and may restore one.
5. *No reference at scale.* The exact engine runs out at moderate treewidth, so beyond it
   an approximate engine's error can only be estimated indirectly.

**Changing the market instead of the algorithm.** Capping how many claims a counterpoint
may bear on, or pricing each NAND with its own small market maker and removing arbitrage
between neighbours (the constraint-generation approach in `literature.md`), would bound
the cost by construction. Both change what traders can express.

## 2. Stress testing with many language-model agents

The properties tested here are properties of the pricing rule. Whether a market built on
it produces good prices depends on how participants behave, and that has not been tested.
The plan is to run many language-model agents, each with private information and its own
goals, against a market on a question with a known answer, and to measure:

- whether prices converge to the answer the agents' pooled information supports;
- whether agents build argument structure that is informative, or flood the market with
  points;
- whether an agent instructed to exploit an approximate engine's pricing errors can find
  and profit from them, which is the empirical counterpart of items 1 and 2 above;
- what argument graphs agents actually build, since the width of those graphs is what
  decides whether exact pricing is enough.

None of this has been run. It needs a harness around the engines here: accounts, a way for
agents to add points and trade, and a scoring of outcomes.
