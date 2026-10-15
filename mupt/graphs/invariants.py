"""
Properties of graphs which depend only on their network structure
(i.e. not on a labelling or embedding of a graph), and logical
predicates derived from such graph properties
"""

from typing import Generator, TypeVar
from collections import Counter

from networkx.classes import Graph

GT = TypeVar("GT", bound=Graph)


def is_indiscrete(graph: Graph) -> bool:
    """
    Whether the current graph has an indiscrete incidence topology,
    in which no nodes are distinguishable by connectivity alone
    """
    return graph.number_of_edges() == 0


# the indiscrete topology is also the trivial one, since it endows
# no further structure than the axioms of a topological space guarantee,
# namely, that the whole set and the mepty set are part of its topology
is_trivial = is_indiscrete


def is_empty(graph: Graph) -> bool:
    """
    Whether the graph is empty (i.e. has no nodes)

    Represents a valid topology, since it contains the empty set
    and itself (which just so happens to also be the empty set)
    """
    return graph.number_of_nodes() == 0


def is_unbranched(graph: Graph) -> bool:
    """
    Whether EVERY connected component of a graph is either
    an unbranching path(s) or an isolated node
    """
    for node, node_degree in graph.degree:
        if node_degree > 2:
            return False
    else:
        return True


is_linear = is_unbranched


def is_branched(graph: Graph) -> bool:
    """Whether the topology contains any branching nodes"""
    return not is_unbranched(graph)


# Node or edge subsets and induced subgraphs
def termini(graph: Graph) -> Generator[int, None, None]:
    """Generate the all nodes of a graph which are incident to exactly one edge"""
    for node, node_degree in graph.degree:
        if node_degree == 1:
            yield node


leaves = termini


# TB: strictly-speaking, this isn't an invariant since it relies on a graph coloring
# consider moving to dedicated submodule if impl grows more complex in the future
def canonical_form(graph) -> str:
    """
    Return a canonical form based on a graph structure and assigned coloring
    Two graphs will have the same canonical forms IFF they are isomorphic
    """
    # return nx.weisfeiler_lehman_graph_hash(graph)
    return str(
        hash(
            # temporary, quick-to-compute stand-in
            # for eventual "real-deal" canonical form
            tuple(Counter(node_degree for node, node_degree in graph.degree).items())
        )
    )
