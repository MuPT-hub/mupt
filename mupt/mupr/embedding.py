'''Utilities for inserting Primitives into pre-built graphs to create polymer topology graphs'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

# DEVNOTE: this is not a submodule under topology to avoid circular imports
# and to shelter MID Graphs from needing to know about HOW they're embedded

from typing import Mapping, Hashable, TypeVar
Label = TypeVar('Label', bound=Hashable)

import networkx as nx

from .primitives import Primitive
from .topology import PolymerTopologyGraph


def embed_primitive_topology(topology : nx.Graph, mapping : Mapping[Label, Primitive]) -> PolymerTopologyGraph:
    '''Embed a labelled lexicon of Primitives into a correspondingly-labelled connectivity graph'''
    if not isinstance(topology, nx.Graph):
        raise TypeError(f'Topology must be a Graph instance, not one of type {type(topology)}')
    
    if not set(topology.nodes).issubset(set(mapping.keys())):
        raise ValueError('Topology node labels must be a subset of the lexicon mapping labels')
    
    # TODO: verify uniqueness of Primitives according to their canonical forms

    # DEVNOTE: copy=False raises errors when attempting to compare Pritive to other types
    embedded_topology = PolymerTopologyGraph(nx.relabel_nodes(topology, mapping=mapping, copy=True))
    for primitive in embedded_topology.nodes:
        # NOTE: valid primitives can however have functionality greater than the node degree IFF external ports are part of the Primitive
        if (degree := embedded_topology.degree(primitive)) > primitive.functionality: 
            raise ValueError(f'Degree {degree} node cannot be embedded with {primitive.functionality}-functional Primitive {primitive!r}')

    # TODO: extract and perform balance check on external primitives vs edges (2*|E| + |X| = sum{I_v, v in V})

    return embedded_topology
