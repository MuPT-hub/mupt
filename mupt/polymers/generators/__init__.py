"""
Topology generators: recipes producing descriptive graphs of polymer chains

    base   : TopologyGenerator, and CompositeGenerator for generators built from components
    linear : LinearPolymerGenerator, single-strand chains
    ladder : LadderPolymerGenerator, double-strand chains
    unit   : UnitGenerator, a single unit (a composite's bare SMILES components)
    block  : BlockCopolymerGenerator, components joined end to end
    star   : StarPolymerGenerator, arms on the free faces of a core
"""

from .base import (
    TopologyGenerator as TopologyGenerator,
    CompositeGenerator as CompositeGenerator,
)
from .linear import LinearPolymerGenerator as LinearPolymerGenerator
from .ladder import LadderPolymerGenerator as LadderPolymerGenerator
from .unit import UnitGenerator as UnitGenerator
from .block import BlockCopolymerGenerator as BlockCopolymerGenerator
from .star import StarPolymerGenerator as StarPolymerGenerator
