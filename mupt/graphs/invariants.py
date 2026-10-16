"""
Properties of graphs which depend only on their network structure
(i.e. not on a labelling or embedding of a graph), and logical
predicates derived from such graph properties
"""

from typing import Generator
from collections import Counter

from networkx.classes import Graph
from networkx.algorithms import connected_components, simple_cycles

from .graphtypes import GraphLike


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
    Whether the graph contains NO branching nodes

    Every connected component of such a graph is either
    a path, a cycle, or an isolated node
    """
    return all(node_degree <= 2 for node, node_degree in graph.degree)


def is_branched(graph: Graph) -> bool:
    """Whether the graph contains ANY branching nodes"""
    return any(node_degree > 2 for node, node_degree in graph.degree)


def is_acyclic(graph: Graph) -> bool:
    """Whether the graph (directed or not) contains any cycles"""
    try:
        _cycle = next(simple_cycles(graph))
    except StopIteration:
        return True
    else:
        return False


def is_linear(graph: Graph) -> bool:
    """
    Whether a graph is unbranched AND acyclic

    All connected components of such a graph are path graphs
    """
    return is_unbranched(graph) and is_acyclic(graph)


# Node or edge subsets and induced subgraphs
def termini(graph: Graph) -> Generator[int, None, None]:
    """Generate the all nodes of a graph which are incident to exactly one edge"""
    for node, node_degree in graph.degree:
        if node_degree <= 1:
            yield node


leaves = termini


def components(graph: GraphLike) -> Generator[GraphLike, None, None]:
    """
    Generates every connected component in the graph as Graph-like instances,
    rather than the node sets that networkx.connected_components provides

    Yielded components will have the same Graph type as the graph passed in
    """
    for cc_nodes in connected_components(graph):
        yield graph.subgraph(cc_nodes)


chains = components


# TB: strictly-speaking, this isn't an invariant since it relies on a graph coloring
# consider moving to dedicated submodule if impl grows more complex in the future
def canonical_form_graph(graph: Graph) -> str:
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
