"""Carroll Markets: a market maker over points tied by NANDs.

    ExactEngine              brute force over every admissible assignment; the ground truth
    JunctionTreeEngine       exact, loops included; cost set by how tangled the graph is
    BeliefPropagationEngine  approximate on loops, linear cost; experimental
    ClusterBPEngine          the same with tightly tied NANDs priced exactly; experimental
"""
from .engine import Engine, ExactEngine, Nand
from .jtree import JunctionTreeEngine, TooTangled, width_of
from .bp import BeliefPropagationEngine
from .cbp import ClusterBPEngine

__all__ = ['Engine', 'ExactEngine', 'Nand', 'JunctionTreeEngine', 'TooTangled', 'width_of',
           'BeliefPropagationEngine', 'ClusterBPEngine']
