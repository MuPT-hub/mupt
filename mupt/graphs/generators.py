"""Routines for creating graph from simplified inputs and deterministic rules"""

from typing import (
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Optional,
)

from itertools import count
from functools import reduce

from networkx.classes import Graph, DiGraph
from networkx.generators import path_graph, balanced_tree
from networkx.algorithms import union as graph_union


def path_graphs(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[Graph] = Graph,
) -> Generator[Graph, None, None]:
    """
    Generate a sequence of path graphs according to a
    provided sequence of lengths and labelling scheme
    """
    if node_labels is None:
        node_labels = count(start=0, step=1)

    for chain_length in chain_lengths:
        yield path_graph(
            (next(node_labels) for _ in range(chain_length)),
            create_using=create_using,
        )


def noodle_graph(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[Graph] = Graph,
) -> Graph:
    """
    Generate a single topology representing a collection of disjoint linear
    chains according to a provided sequence of lengths and labelling scheme
    """
    return reduce(
        graph_union,
        path_graphs(
            chain_lengths=chain_lengths,
            node_labels=node_labels,
            create_using=create_using,
        ),
    )


from typing import Hashable


def balanced_dendimer_graph(
    core: Hashable = "core",
    shell_inner: Hashable = "branch",
    shell_outer: Hashable = "terminus",
    coord_number: int = 3,
    num_generations: int = 5,
    label_attr_name: str = "label",
) -> Graph:
    """
    Generate a Cayley graph, i.e. a tree whose non-leaf
    nodes all have the same fixed node degree

    Can specify named labels for the core, branching, and leaf nodes

    Useful simplified model for the connectivity of a
    dendrimer molecule at the repeat-unit level
    """
    dendr_tree = balanced_tree(
        coord_number,
        num_generations,
        create_using=DiGraph,
    )
    for node in dendr_tree.nodes:
        if dendr_tree.in_degree(node) == 0:
            dendr_tree.nodes[node][label_attr_name] = core
        elif dendr_tree.out_degree(node) == 0:
            dendr_tree.nodes[node][label_attr_name] = shell_outer
        else:
            dendr_tree.nodes[node][label_attr_name] = shell_inner
    return Graph(dendr_tree)  # make edges undirected


cayley_graph = balanced_dendimer_graph
