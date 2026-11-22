# Carroll Markets

Prediction markets that show their reasoning. An ordinary prediction market gives you a
probability and asks you to trust it. A Carroll Market also prices the arguments: every
claim, every counterpoint raised against it, and the question of whether that counterpoint
actually bears on the claim are tradable assets, priced together by one market maker so
that the prices cannot contradict one another. Reading the market tells you what
the crowd believes and which arguments are carrying that belief.

This repository is the market maker: the pricing rule, exact reference implementations,
and experimental fast approximations, with tests. It has no accounts, no user interface
and no incentive layer on top of prices.

## Status

Research code, pure Python 3 (3.9+), no dependencies.

| engine | what it is | state |
|---|---|---|
| `ExactEngine` | brute force over every admissible assignment | the ground truth; up to 24 points |
| `JunctionTreeEngine` | exact, loops included; cost grows with how tangled the graph is | working; use this one |
| `BeliefPropagationEngine` | approximate on loops, cost linear in graph size | experimental |
| `ClusterBPEngine` | the same, with tightly tied arguments priced exactly | experimental |

The two experimental engines are not safe to run a real market on. Their prices are close
to exact on the graphs we have tested, but the small errors that remain lean in a
consistent direction, and we have not yet measured what that is worth to a trader who
knows about it. See `docs/open-problems.md`.

## Quickstart

```bash
git clone https://github.com/connormcmk/carroll-markets.git && cd carroll-markets
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 examples/worked_market.py
```

Or install it (`pip install -e .`) and drop the `PYTHONPATH=src`.

## A minimal example

A claim, one counterpoint, and the relevance point that ties them. The tie is a NAND: the
claim, the counterpoint and the relevance cannot all be true.

```python
from carroll_markets import JunctionTreeEngine

market = JunctionTreeEngine(
    points=['claim', 'counterpoint', 'relevance'],
    nands=[('claim', 'counterpoint', 'relevance')],
    b=100.0,                       # depth: larger means prices move less per share
)
market.prices()                    # each 0.4286: seven assignments are admissible, three have any one point true

market.trade({'counterpoint': 80.0})   # buy 80 shares; returns the cost, 42.21
market.prices()                    # counterpoint 0.625, claim 0.396, relevance 0.396

market.trade({'relevance': 80.0})  # someone backs the relevance; cost 39.54
market.prices()                    # counterpoint 0.593, relevance 0.593, claim 0.344
```

The argument bites to the extent the market believes both that the counterpoint is true
and that it matters. Here all three points were created together, so the relevance starts
at 0.43 and buying the counterpoint alone already lowers the claim. In a live market a new
relevance point enters near zero, and a counterpoint moves its claim only once someone
backs the relevance. `examples/worked_market.py` shows that, with two competing
counterpoints and a settlement.

## What is here

```
src/carroll_markets/
    engine.py    the engine interface and ExactEngine
    jtree.py     JunctionTreeEngine
    bp.py        BeliefPropagationEngine
    cbp.py       ClusterBPEngine
    oracle.py    test graph families, and measures of whether an engine's errors can be exploited
    bench.py     how an engine's cost grows with graph size
    survey.py    how tangled large graphs get, and where exact pricing stops
tests/           what every engine must satisfy, and one file per engine
examples/        the worked market above
docs/
    mechanism.md      plain-language introduction, then the formal model
    literature.md     related work and how this differs
    open-problems.md  what is not solved
```

## Writeup

`docs/mechanism.md` is the writeup: a plain-language introduction followed by the formal
model. `docs/literature.md` places it in prior work and `docs/open-problems.md` says what
is not solved.

## Contributing

This repository is generated from the authors' working repository, so changes made here
directly are overwritten at the next release. Please open an issue instead of a pull
request.

## License

MIT. See `LICENSE`.
