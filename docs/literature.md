# Related work

Every reference below was checked against a publisher, proceedings or preprint listing.
Anything marked TODO could not be confirmed and should not be cited until it is.

## Market scoring rules

Hanson introduced the market scoring rule, in which a market maker always quotes a price
and each trader is paid by a proper scoring rule for the change they make to the market's
probabilities. The logarithmic version (LMSR) is the cost function used here. Hanson's
papers already frame the goal as eliciting a joint distribution over many related
variables, and show that the logarithmic rule preserves conditional independence
relations when one variable is traded.

- R. Hanson. Combinatorial Information Market Design. *Information Systems Frontiers*
  5(1):107–119, 2003. doi:10.1023/A:1022058209073
- R. Hanson. Logarithmic Market Scoring Rules for Modular Combinatorial Information
  Aggregation. *Journal of Prediction Markets* 1(1):3–15, 2007.

## Hardness of combinatorial market making

Pricing an LMSR market maker over a combinatorial outcome space means summing over
exponentially many outcomes. Chen et al. show that exact pricing is computationally hard
for the betting languages they study (bets on permutations and on Boolean combinations
of events). Chen, Goel and Pennock give an early tractable special case, for
single-elimination tournaments, by representing prices as a Bayesian network.

- Y. Chen, L. Fortnow, N. Lambert, D. M. Pennock, J. Wortman. Complexity of Combinatorial
  Market Makers. *Proceedings of the 9th ACM Conference on Electronic Commerce (EC '08)*,
  190–199, 2008. doi:10.1145/1386790.1386822. arXiv:0802.1362
- Y. Chen, S. Goel, D. M. Pennock. Pricing Combinatorial Markets for Tournaments.
  *Proceedings of the 40th ACM Symposium on Theory of Computing (STOC '08)*, 305–314, 2008.

## Tractable combinatorial market makers

Three lines of work make large combinatorial markets practical.

*Convex cost functions over a relaxed price space.* Abernethy, Chen and Vaughan
characterise market makers by convex cost functions built through conjugate duality, and
show that relaxing the set of allowed prices can restore tractability while keeping the
market maker's loss bounded.

- J. Abernethy, Y. Chen, J. Wortman Vaughan. Efficient Market Making via Convex
  Optimization, and a Connection to Online Learning. *ACM Transactions on Economics and
  Computation* 1(2), Article 12, 2013. doi:10.1145/2465769.2465777

*Independent markets plus arbitrage removal.* Dudík, Lahaie and Pennock price related
securities with separate market makers and remove the arbitrage between them by
generating constraints as needed. Kroer et al. extend this with an integer-programming
oracle.

- M. Dudík, S. Lahaie, D. M. Pennock. A Tractable Combinatorial Market Maker Using
  Constraint Generation. *Proceedings of the 13th ACM Conference on Electronic Commerce
  (EC '12)*, 2012. doi:10.1145/2229012.2229047
- C. Kroer, M. Dudík, S. Lahaie, S. Balakrishnan. Arbitrage-Free Combinatorial Market
  Making via Integer Programming. *Proceedings of the 2016 ACM Conference on Economics
  and Computation (EC '16)*, 161–178, 2016. doi:10.1145/2940716.2940767. arXiv:1606.02825

*Graphical-model inference.* Sun et al. represent the market's joint distribution as a
Bayesian network and extend the junction tree algorithm to update prices and track
traders' assets. This is the closest prior work to the exact engine here.

- W. Sun, R. Hanson, K. B. Laskey, C. Twardy. Probability and Asset Updating using
  Bayesian Networks for Combinatorial Prediction Markets. *Proceedings of the 28th
  Conference on Uncertainty in Artificial Intelligence (UAI 2012)*, 2012. arXiv:1210.4900

## Inference algorithms used here

The exact engine is the junction tree method. The approximate engines are loopy belief
propagation with the Bethe free energy as the cost function, and a small step toward
generalised belief propagation, in which groups of tightly coupled factors are treated
exactly. Tree-reweighted bounds are relevant because they give an upper bound on the log
partition function, which the Bethe approximation does not.

- S. L. Lauritzen, D. J. Spiegelhalter. Local Computations with Probabilities on Graphical
  Structures and Their Application to Expert Systems. *Journal of the Royal Statistical
  Society, Series B* 50(2):157–224, 1988.
- J. S. Yedidia, W. T. Freeman, Y. Weiss. Constructing Free-Energy Approximations and
  Generalized Belief Propagation Algorithms. *IEEE Transactions on Information Theory*
  51(7):2282–2312, 2005. doi:10.1109/TIT.2005.850085
- M. J. Wainwright, T. S. Jaakkola, A. S. Willsky. A New Class of Upper Bounds on the Log
  Partition Function. *IEEE Transactions on Information Theory* 51(7):2313–2335, 2005.
  doi:10.1109/TIT.2005.850091

## How Carroll Markets differ

- **The structure is the argument.** In prior combinatorial markets the outcome space is
  fixed by the designer (an election map, a tournament bracket, a Bayesian network drawn
  in advance). Here the structure is built by participants as they argue: each new
  counterpoint adds a point, a relevance point and one constraint.
- **Relevance is itself traded.** Whether a counterpoint bears on a claim is not asserted
  by whoever raises it. It is a point with a price, and it can be argued against in turn.
- **One kind of security, one kind of constraint.** Traders buy and sell single points
  only, and every constraint is a NAND over three points. The market state is one number
  per point, and pricing is inference in a model with unary fields and hard ternary
  factors. That narrowness is what makes local algorithms plausible.
- **A different approximation question.** Constraint-generation market makers keep
  securities separate and remove arbitrage between them. The approximate engines here keep
  one cost function (so trades stay path independent) and approximate the partition
  function instead. Whether that can be made safe against informed traders is open; see
  `open-problems.md`.

We do not have a hardness result specific to NAND-triple markets. The exact engines here
cost time exponential in the number of points (`ExactEngine`) or in the treewidth
(`JunctionTreeEngine`).
