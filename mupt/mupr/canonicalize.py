'''Implementations of graph canonicalization, given a graph with an ordered coloring'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Hashable, Iterable, Protocol, runtime_checkable
from collections import Counter

import networkx as nx


@runtime_checkable
class Canonicalizable(Protocol):
    '''Object with a notion of a canonical representative of instances which are equivalent in some sense'''
    def canonical_form(self) -> Hashable:
        ...

# graphs
def canonical_graph_property(graph : nx.Graph) -> str:
    '''Canonicalize a graph with an ordered coloring'''
    raise NotImplementedError('Graph canonicalization is not implemented yet')

# multisets
def lex_order_multiset(elements : Iterable[Hashable]) -> tuple[tuple[Hashable, int], ...]:
    '''
    Generate a lexicographically-ordered presentation of a multiset of elements
    
    Returns as a tuple of (element, count) pairs, where "ties" on any ordering 
    by elements are broken by the totally-ordered counts
    '''
    return tuple(sorted(Counter(elements).items()))

def lex_order_multiset_str(
        elements : Iterable[Hashable],
        element_repr : Callable[[Hashable], str]=str,
        separator : str=':',
        joiner : str='-',
    ) -> str:
    '''
    Generate a lexicographically-ordered presentation of a multiset of elements as a string
    
    Returns as a string of (element, count) pairs, where "ties" on any ordering 
    by elements are broken by the totally-ordered counts
    '''
    return joiner.join(
        f'{element_repr(element)}{separator}{count}'
            for (element, count) in lex_order_multiset(elements)
    )