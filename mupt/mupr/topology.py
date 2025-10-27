'''Monomer Interconnectivity and Degree (MID) graphs, for encoding the topological connectivity of a polymer system'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    TypeAlias,
)
from itertools import count
from functools import reduce
from collections import Counter

from numpy import ndarray
import networkx as nx
from matplotlib.axes import Axes


GraphLayout : TypeAlias = Callable[[nx.Graph], dict[Hashable, ndarray]]
    
class TopologicalStructure(nx.Graph): 
    # DEV: opting not to call this just "Topology" for now to avoid confusion, 
    # ...since many molecular packages also have a class by that name
    '''
    An incidence topology induces on a set of Primitives,
    Represented as a Graph whose edge pairs generate the topology
    '''
    # network properties
    @property
    def is_indiscrete(self) -> bool:
        '''Whether the current topology represents an indiscrete topology (i.e. a "trivial topology", one without connections)'''
        return self.number_of_edges() == 0
    is_trivial = is_indiscrete
    
    @property
    def is_empty(self) -> bool:
        '''
        Whether the topology is empty (i.e. has no nodes)
        
        Represents a valid topology, since it contains the empty set
        and itself (which just so happens to also be the empty set)
        '''
        return self.number_of_nodes() == 0

    @property
    def is_unbranched(self) -> bool:
        '''Whether the topology contains only unbranching chain(s) or isolated nodes'''
        return all(node_deg <= 2 for node_id, node_deg in self.degree)
    is_linear = is_unbranched

    @property
    def is_branched(self) -> bool:
        '''Whether the topology contains any branching nodes'''
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
        '''The number of disconnected chains represented within the topology'''
        return nx.number_connected_components(self)

    @property
    def chains(self) -> Generator['TopologicalStructure', None, None]:
        '''Generates all disconnected polymers chains in the graph sequentially'''
        for cc_nodes in nx.connected_components(self):
            yield TopologicalStructure(self.subgraph(cc_nodes))
            
    # depiction
    def canonical_form(self) -> str:
        '''
        Return a canonical form based on the graph structure and coloring iduced by the canonical forms of internal Primitives
        Tantamount to solving the graph isomorphism problem 
        '''
        # raise NotImplementedError('Graph canonicalization is not implemented yet')
        # return nx.weisfeiler_lehman_graph_hash(self) # stand-in for more specific implementation to follow
        return hash(tuple(Counter(deg for node, deg in self.degree).items())) # temporary, quick-to-compute stand-in for eventual "real-deal" canonical form

    def __repr__(self) -> str:
        #TODO: make this more descriptive
        # return super().__repr__()
        return f'{self.__class__.__name__}(num_objects={self.number_of_nodes()}, indiscrete={self.is_indiscrete})'
    
    def visualize(
        self,
        ax : Optional[Axes]=None,
        layout : GraphLayout=nx.kamada_kawai_layout,
        **draw_kwargs,
    ) -> None:
        '''
        Draw the topology's graph
        '''
        if 'with_labels' not in draw_kwargs:
            draw_kwargs['with_labels'] = True
        
        nx.draw(
            self,
            ax=ax,
            pos=layout(self),
            **draw_kwargs,
        )

# graph generators
def path_graphs(
    chain_lengths : Iterable[int],
    node_labels : Optional[Iterator[Hashable]]=None,
    create_using : type[nx.Graph]=TopologicalStructure,
) -> Generator[nx.Graph, None, None]:
    '''
    Generate a sequence of path graphs according to a provided sequence of lengths and labelling scheme
    '''
    if node_labels is None:
        node_labels = count(start=0, step=1)
        
    for chain_length in chain_lengths:
        yield nx.path_graph(
            (next(node_labels) for _ in range(chain_length)),
            create_using=create_using,
        )
        
def noodle_graph(
    chain_lengths : Iterable[int],
    node_labels : Optional[Iterator[Hashable]]=None,
    create_using : type[nx.Graph]=TopologicalStructure,
) -> nx.Graph:
    '''
    Generate a single topology representing a collection of disjoint linear
    chains according to a provided sequence of lengths and labelling scheme
    '''
    return reduce(
        nx.union,
        path_graphs(
            chain_lengths=chain_lengths,
            node_labels=node_labels,
            create_using=create_using,
        )
    )