'''Monomer Interconnectivity and Degree (MID) graphs, for encoding connectivity of a polymer system'''

from typing import Generator
import networkx as nx

from ..primitives import MolecularPrimitive, PrimitiveLexicon


class PolymerTopologyGraph(nx.Graph):
    '''A graph representation of the connectivity of monomer fragments in a polymer topology'''

    # network properties
    @property
    def num_monomers(self) -> int:
        '''Number of monomer units represented in the current polymer'''
        return self.number_of_nodes()
    DOP = num_monomers

    @property
    def is_unbranched(self) -> bool:
        '''Whether the monomer graph represents straight chain(s) without branching'''
        return all(node_deg <= 2 for node_id, node_deg in self.degree)
    is_linear = is_unbranched

    @property
    def is_unbranched(self) -> bool:
        '''Whether the monomer graph represents straight chain(s) without branching'''
        return not self.is_unbranched
    
    @property
    def terminal_monomers(self) -> Generator[int, None, None]:
        '''Generates the indices of all nodes corresponding to terminal monomers (i.e. those wiht only one outgoing bond)'''
        for node_idx, degree in self.degree:
            if degree == 1:
                yield node_idx
    termini = leaves = terminal_monomers
    
    @property
    def num_chains(self) -> int:
        '''The number of disconnected chains represented by the MonoGraph'''
        return nx.number_connected_components(self)

    @property
    def chains(self) -> Generator['PolymerTopologyGraph', None, None]:
        '''Generates all disconnected polymers chains in the graph sequentially'''
        for cc_nodes in nx.connected_components(self):
            yield self.subgraph(cc_nodes)
            
            
    # primitive handlers
    def insert_primitives(self, lexicon : PrimitiveLexicon) -> None:
        '''Map infomration from primitive lexicon basis set into graph topology'''
        raise NotImplemented

    def _validate(self, lexicon : PrimitiveLexicon) -> bool:
        '''Check whether a primitive lexicon provided is compatible with a graph'''
        raise NotImplemented

MonomerInterconnectivityAndDegreeGraph = MIDGraph = PolymerTopologyGraph # aliases for convenience