"""A small worked market: one claim, two competing arguments against it.

    python3 examples/worked_market.py

The claim is that a bridge should reopen this month. Two counterpoints are
raised against it while the market is live. Each is tied to the claim through
its own relevance point, which says "if this counterpoint is true, the claim
fails". A new relevance point enters at a price near zero: raising a
counterpoint costs the claim nothing until someone stakes money on the
counterpoint mattering.
"""
from carroll_markets import JunctionTreeEngine

MEANING = {
    'reopen':         'The bridge should reopen this month',
    'crack':          'The inspection will find a crack in a girder',
    'crack_matters':  'A cracked girder means the bridge should not reopen',
    'budget':         'The repair budget will not be approved',
    'budget_matters': 'Without the repair budget the bridge should not reopen',
}


def run(verbose=True):
    market = JunctionTreeEngine(['reopen'], b=100.0)
    steps = []

    def step(title):
        prices = market.prices()
        steps.append((title, prices))
        if verbose:
            print(f'\n{title}')
            for name in MEANING:
                if name in prices:
                    print(f'  {name:<15} {prices[name]:.3f}')

    step('The market opens with the claim alone')

    paid = market.trade({'reopen': 120.0})
    step(f'A trader buys 120 shares of the claim, paying {paid:.2f}')

    market.add_point('crack')
    market.add_relevance('reopen', 'crack', 'crack_matters')    # claim, counterpoint, relevance: not all three
    paid = market.trade({'crack': 150.0})
    step(f'A counterpoint is raised and 150 shares of it bought for {paid:.2f}. Nobody has backed its relevance')

    shares = market.shares_for_spend('crack_matters', 60.0)
    market.trade({'crack_matters': shares})
    step('Someone spends 60 on that relevance: the argument now bites')

    market.add_point('budget')
    market.add_relevance('reopen', 'budget', 'budget_matters')
    paid = market.trade({'budget': 150.0})
    step(f'A second counterpoint is raised and bought for {paid:.2f}. Its relevance stays unbacked')

    market.settle('crack', False)
    step('The inspection finds no crack: the first counterpoint settles false')
    return steps


if __name__ == '__main__':
    run()
