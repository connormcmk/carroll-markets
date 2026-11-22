# How a Carroll Market works

The first half of this document is a plain-language introduction. The second half states
the model formally. Every number quoted here is checked by `tests/test_docs.py`.

## Part 1: in plain language

### The trust gap

A prediction market turns many people's beliefs into one number. "The bridge reopens this
month: 65%." The number is often good, but it comes with no reasons. A reader who wants to
know why it is 65%, or what would have to change for it to fall to 30%, has nothing to go
on except trust in the crowd.

A discussion forum has the opposite problem. It is full of reasons, and nothing says
which of them anyone would stake money on.

A Carroll Market closes the gap by pricing the reasons themselves.

### Arguments become priced assets

Everything in the market is a **point**: a statement that can be true or false, with
shares that pay 1 if it is true and 0 if it is false. A point's price is the market's
probability that it is true.

An argument against a claim is made of two further points:

- a **counterpoint**: a statement that, if true, counts against the claim;
- a **relevance** point: the statement that this counterpoint really does defeat this claim.

So "the inspection will find a crack" is a counterpoint to "the bridge should reopen", and
"a cracked girder means the bridge should not reopen" is the relevance point between them.
Each of the three has its own price. Someone who thinks the crack is real but harmless
buys the counterpoint and leaves the relevance alone. Someone who thinks the crack would
be decisive but does not exist does the opposite.

A relevance point is an ordinary point, so it can have counterpoints of its own ("the
girder is redundant, so a crack does not matter"). Arguments about arguments nest to any
depth, and one point can serve as a counterpoint in several places.

### The NAND relationship

The three points of an argument are tied by one rule: **they cannot all be true.** If the
counterpoint is true, and it really does defeat the claim, then the claim is false. In
logic this is a NAND over the three.

The market maker enforces the rule by refusing to consider any state of the world in
which all three hold. Nothing else is imposed. Any two of them may be true together, and
all three may be false.

### How an update propagates

Because the market maker prices all points together, a trade in one point moves the
others. `examples/worked_market.py` shows it on a claim with two counterpoints, raised one
after the other while the market is live. The claim has been bought up to 0.769:

| what happens | claim's price |
|---|---|
| a counterpoint is raised and bought up to 0.82; nobody backs its relevance | 0.767 |
| someone spends 60 on that relevance, taking it to 0.45 | 0.591 |
| a second counterpoint is raised and bought up to 0.82; its relevance stays unbacked | 0.589 |
| the first counterpoint is settled false | 0.767 |

A counterpoint alone does almost nothing to the claim, however strongly it is believed. A
new relevance point enters at a price near zero, and until someone stakes money on the
counterpoint mattering, the market has no reason to think it does. Once the relevance is
backed, the claim falls: the argument bites to the extent the market believes both that
the counterpoint is true and that it defeats the claim. When the world answers a point
(here, the inspection finds no crack), its price is fixed, every other price becomes the
probability given that answer, and the claim is released.

The pressure runs in every direction. Buying the claim pushes its counterpoints and their
relevance points down, since if the claim is true then, for each argument against it,
either the counterpoint is false or it does not matter.

### Why prices are honest: proper scoring

The market maker is a logarithmic market scoring rule. Its defining property is that a
trader does best, in expectation, by moving prices to what they actually believe. A trader
who thinks a price is too low profits by buying until it matches their belief, and loses by
pushing it past that. The same holds for every point at once: the most profitable trade
moves each point's price to the trader's own probability for that point.

Two further properties follow from the same rule. There is always a price to trade at, so
a market with one participant still works. And the market maker's worst possible loss is
fixed in advance by one number, the depth `b`.

### What reading the market gives you

For any claim you can read its price, the price of every counterpoint raised against it,
and the price of each relevance. A claim at 40% with a counterpoint at 70% and that
counterpoint's relevance at 65% tells you what is holding the claim down. The same claim
at 40% with every counterpoint low tells you the doubt is about the claim itself.

## Part 2: the formal model

### Points, NANDs, admissible assignments

Let `V = {1, ..., n}` be the points. An assignment is `x ∈ {0,1}^n`, where `x_i = 1`
means point `i` is true.

Let `T` be a set of ordered triples `(a, c, r)` of distinct points: a claim, a
counterpoint and a relevance. The admissible assignments are

    Ω = { x ∈ {0,1}^n :  x_a · x_c · x_r = 0  for every (a, c, r) ∈ T }.

### The cost function

The market's state is a vector `s ∈ R^n`: the net shares of each point held by traders.
Each point also has a fixed starting weight `o_i`, a real number that sets the price it
enters at (zero for one half; see "Adding points to a live market" below). Write `θ = (s + o) / b`,
with depth `b > 0`. The cost function is

    C(s) = b · log Z(s),        Z(s) = Σ_{x ∈ Ω} exp(θ · x).

A trade `δ ∈ R^n` (positive entries buy, negative sell) costs `C(s + δ) − C(s)`.

### Prices

The market's distribution over assignments is

    P_s(x) = exp(θ · x) / Z(s)    for x ∈ Ω,    and 0 otherwise.

The price of point `i` is the marginal cost of its shares, which is its probability:

    p_i(s) = ∂C/∂s_i = P_s(x_i = 1).

This is an exponential family with one sufficient statistic per point and hard
constraints over triples: a Markov random field with unary fields and ternary factors.
Pricing is inference in that model. Computing `C` is computing a log partition function,
and computing prices is computing marginals.

### Properties (each is a test in `tests/test_engine.py`)

- **Path independence.** The cost of a sequence of trades depends only on the start and
  end states. A round trip is free, and the order of trades does not change the total.
- **Prices are probabilities**, strictly between 0 and 1 until settlement.
- **Coherence.** For every triple, `p_a + p_c + p_r ≤ 2`, since `x_a + x_c + x_r ≤ 2` on
  every admissible assignment.
- **No arbitrage.** No trade pays more than it costs in every admissible assignment.
- **Bounded loss.** With all starting weights zero, whatever is traded and however the
  world turns out, the market maker loses at most `b · log |Ω|`.
- **Convexity.** `C` is convex, so each further share of a point costs more than the last.

### Settlement

Settling point `i` to value `v` restricts `Ω` to assignments with `x_i = v`. Each share
of `i` pays `v`. All other prices become conditional probabilities given `x_i = v`. A
settlement that leaves no admissible assignment is rejected.

### Proper scoring

Suppose a risk-neutral trader holds beliefs `q` over `Ω`, with marginals
`q_i = q(x_i = 1)`. The expected profit of a trade `δ` is

    Σ_i δ_i · q_i − ( C(s + δ) − C(s) ).

This is concave in `δ`, and its gradient is `q − p(s + δ)`. It is maximised exactly when
the prices after the trade equal the trader's marginals. Traders trade only in points, not
in arbitrary combinations, so a trader can express a belief about each point and the
structure `T` supplies the dependence between them.

### Adding points to a live market

A point's starting weight sets the price it opens at. It is not a price: prices are
always between 0 and 1. The weight is on the log-odds scale, in share units, so a point
standing alone with weight `o` and no shares sold opens at `1 / (1 + exp(−o / b))`. Zero
gives one half, a negative weight gives a price below one half, a positive one above.

A new unattached point enters with weight zero, at price one half, and moves no other
price. A new relevance point enters at a low price (`add_relevance`, default 0.01, which
is a weight of `b · ln(0.01 / 0.99)`, about −4.6 `b`), so that tying it to its claim and
counterpoint moves their prices very little. Its price then rises only as traders buy it.

### Computation

`Ω` can have up to `2^n` members, so summing over it directly (`ExactEngine`) is feasible
only for small markets. The other engines compute the same quantities differently:

- `JunctionTreeEngine` eliminates points one at a time and passes tables between groups
  of neighbouring points. It is exact for any graph. Its cost is exponential only in the
  size of the largest group (the graph's treewidth under the elimination order used), not
  in `n`.
- `BeliefPropagationEngine` passes messages between points and NANDs until they settle,
  and uses the Bethe approximation to `log Z` as the cost function. It is exact when the
  graph has no loops. Its cost per sweep is linear in the number of NANDs.
- `ClusterBPEngine` first merges NANDs that share two points into clusters that are summed
  exactly, then passes messages between clusters.

`docs/open-problems.md` says what is known about the error of the last two.
