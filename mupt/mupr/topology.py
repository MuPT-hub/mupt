"""
Utilities for graphs which encode neighbor connectivity
of chemical systems in the MuPT molecular representation
"""

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

GraphLayout: TypeAlias = Callable[[nx.Graph], dict[Hashable, ndarray]]


# Network properties
def is_indiscrete(graph: nx.Graph) -> bool:
    """Whether the current topology represents an indiscrete topology
    i.e. a "trivial topology" without any connections
    """
    return graph.number_of_edges() == 0


is_trivial = is_indiscrete


def is_empty(graph: nx.Graph) -> bool:
    """
    Whether the topology is empty (i.e. has no nodes)

    Represents a valid topology, since it contains the empty set
    and itself (which just so happens to also be the empty set)
    """
    return graph.number_of_nodes() == 0


def is_unbranched(graph: nx.Graph) -> bool:
    """Whether the topology contains only unbranching chain(s) or isolated nodes"""
    return all(node_deg <= 2 for node_id, node_deg in graph.degree)


is_linear = is_unbranched


def is_branched(graph: nx.Graph) -> bool:
    """Whether the topology contains any branching nodes"""
    return not is_unbranched(graph)


def termini(graph: nx.Graph) -> Generator[int, None, None]:
    """
    Generates the indices of all nodes corresponding to terminal primitives
    (i.e. those with only one outgoing bond)
    """
    for node_idx, degree in graph.degree:
        if degree == 1:
            yield node_idx


leaves = termini


def canonical_graph_property(graph: nx.Graph) -> str:
    """
    Return a canonical form based on the graph structure and coloring
    induced by the canonical forms of internal Primitives
    Tantamount to solving the graph isomorphism problem
    """
    # raise NotImplementedError('Graph canonicalization is not implemented yet')
    # return nx.weisfeiler_lehman_graph_hash(self)
    # # stand-in for more specific implementation to follow

    return str(
        hash(
            # temporary, quick-to-compute stand-in
            # for eventual "real-deal" canonical form
            tuple(Counter(deg for node, deg in graph.degree).items())
        )
    )


# graph generators
def path_graphs(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[nx.Graph] = nx.Graph,
) -> Generator[nx.Graph, None, None]:
    """
    Generate a sequence of path graphs according to a
    provided sequence of lengths and labelling scheme
    """
    if node_labels is None:
        node_labels = count(start=0, step=1)

    for chain_length in chain_lengths:
        yield nx.path_graph(
            (next(node_labels) for _ in range(chain_length)),
            create_using=create_using,
        )


def noodle_graph(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[nx.Graph] = nx.Graph,
) -> nx.Graph:
    """
    Generate a single topology representing a collection of disjoint linear
    chains according to a provided sequence of lengths and labelling scheme
    """
    return reduce(
        nx.union,
        path_graphs(
            chain_lengths=chain_lengths,
            node_labels=node_labels,
            create_using=create_using,
        ),
    )
