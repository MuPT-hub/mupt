'''Monomer Interconnectivity and Degree (MID) graphs, for encoding the topological connectivity of a polymer system'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generator
import networkx as nx

from .structure import Structure
from .primitives import Primitive


class PolymerTopologyGraph(nx.Graph, Structure):
    '''A graph representation of the connectivity of primitives in a polymer topology'''
    # embedding
    ...

    # node properties       
    @property
    def num_atoms(self) -> int:
        '''Number of atoms collectively held within the topology'''
        return sum(primitive.num_atoms for primitive in self.components())
    
    @property
    def is_composite(self) -> bool:
        return True
    
    def _get_components(self) -> Generator[Primitive, None, None]:
        for node in self.nodes:
            yield node
    
    # network properties
    @property
    def is_unbranched(self) -> bool:
        '''Whether the topology represents contains all straight, unbranching chain(s)'''
        return all(node_deg <= 2 for node_id, node_deg in self.degree)
    is_linear = is_unbranched

    @property
    def is_unbranched(self) -> bool:
        '''Whether the topology represents straight chain(s) without branching'''
        return not self.is_unbranched
    
    @property
    def termini(self) -> Generator[int, None, None]:
        '''Generates the indices of all nodes corresponding to terminal primitives (i.e. those with only one outgoing bond)'''
        for node_idx, degree in self.degree:
            if degree == 1:
                yield node_idx
    leaves = termini
    
    @property
    def num_chains(self) -> int:
        '''The number of disconnected chains represented by the MonoGraph'''
        return nx.number_connected_components(self)

    @property
    def chains(self) -> Generator['PolymerTopologyGraph', None, None]:
        '''Generates all disconnected polymers chains in the graph sequentially'''
        for cc_nodes in nx.connected_components(self):
            yield self.subgraph(cc_nodes)
            
    # canonicalization
    def canonical_form(self) -> str:
        '''
        Return a canonical form based on the graph structure and coloring iduced by the canonical forms of internal Primitives
        Tantamount to solving the graph isomorphism problem 
        '''
        return nx.weisfeiler_lehman_graph_hash(self) # stand-in for more specific implementation to follow
        # raise NotImplementedError('Graph canonicalization is not implemented yet')
        

MonomerInterconnectivityAndDegreeGraph = MIDGraph = PolymerTopologyGraph # aliases for convenience