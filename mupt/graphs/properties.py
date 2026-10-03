"""NodePredicates, quantities, and labels which are calculated from graphs"""

from typing import Generator
from collections import Counter

from networkx import Graph


def is_indiscrete(graph: Graph) -> bool:
    """Whether the current topology represents an indiscrete topology
    i.e. a "trivial topology" without any connections
    """
    return graph.number_of_edges() == 0


is_trivial = is_indiscrete


def is_empty(graph: Graph) -> bool:
    """
    Whether the topology is empty (i.e. has no nodes)

    Represents a valid topology, since it contains the empty set
    and itself (which just so happens to also be the empty set)
    """
    return graph.number_of_nodes() == 0


def is_unbranched(graph: Graph) -> bool:
    """Whether the topology contains only unbranching chain(s) or isolated nodes"""
    return all(node_deg <= 2 for node_id, node_deg in graph.degree)


is_linear = is_unbranched


def is_branched(graph: Graph) -> bool:
    """Whether the topology contains any branching nodes"""
    return not is_unbranched(graph)


def termini(graph: Graph) -> Generator[int, None, None]:
    """
    Generates the indices of all nodes corresponding to terminal primitives
    (i.e. those with only one outgoing bond)
    """
    for node_idx, degree in graph.degree:
        if degree == 1:
            yield node_idx


leaves = termini


def canonical_graph_property(graph: Graph) -> str:
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
