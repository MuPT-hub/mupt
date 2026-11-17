from typing import Hashable, Mapping, Literal, Optional
from abc import ABC, abstractmethod

import numpy as np

from networkx import Graph
from rdkit import Chem

from mupt.mupr.primitives import Primitive, PrimitiveHandle, RootPrimitive
from mupt.mupr.connection import ConnectorHandle
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.interfaces.rdkit import primitive_to_rdkit_mols, primitive_to_rdkit
from mupt.polymers._common import as_rng
from mupt.polymers.utils.realize import primitive_from_descriptive_graph
from mupt.polymers.generators.base import TopologyGenerator
from mupt.polymers.generators.linear import LinearPolymerGenerator

#region types
type SeedLike = int | np.random.Generator | None
type LexiconInput = Mapping[Hashable, Primitive | str]
type Lexicon = Mapping[Hashable, Primitive]
type LexiconComposite = Mapping[Hashable, TopologyGenerator]

def merge_universes(universes: list[Primitive]):
    '''
    Merges root primatives, used for combining polymer systems into blends
    '''
    ...

def flatten_lexicon(
        generators: dict[Hashable, TopologyGenerator],
        new_keys: bool = False,
        ) -> dict[Hashable, LinearPolymerGenerator]:
    lexicon = {}
    for name, gen in generators.items():
        key_overlap = set(lexicon.keys()) & set(gen.lexicon.keys())
        if key_overlap:
            raise ValueError(f'Overlapping keys in the lexicons ({key_overlap})')
        lexicon.update


#region generators
class CompositeGenerator(TopologyGenerator, ABC):
    ...

class BlockCopolymerGenerator(TopologyGenerator):

    def __init__(
            self,
            blocks: dict[Hashable, LinearPolymerGenerator],
            ) -> None:
        self.blocks = blocks
        self.lexicon = self._flatten_lexicon(blocks)


class StarCopolymer(CompositeGenerator):
    ...