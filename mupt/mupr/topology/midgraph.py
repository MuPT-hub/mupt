'''Monomer Interconnectivity and Degree (MID) graphs, for encoding connectivity of a polymer system'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generator
import networkx as nx


class PolymerTopologyGraph(nx.Graph):
    '''A graph representation of the connectivity of primitives in a polymer topology'''

    # network properties
    @property
    def num_primitives(self) -> int:
        '''Number of primitive units represented in the current topology'''
        return self.number_of_nodes()
    DOP = num_primitives

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

MonomerInterconnectivityAndDegreeGraph = MIDGraph = PolymerTopologyGraph # aliases for convenience