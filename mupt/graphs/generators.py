"""Routines for creating graphs from simplified non-graph inputs"""

from typing import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Optional,
)

from itertools import count
from functools import reduce

from networkx.classes import Graph
from networkx.generators import path_graph
from networkx.algorithms import union as graph_union

from graphtypes import GraphLike


def path_graph_from_sequence(
    sequence: Collection[Hashable],
    label_attr: str = "label",
    create_using: type[GraphLike] = Graph,
) -> GraphLike:
    """
    Generate a path graph from a sequence of labels

    Keeps identical labels as distinct node,
    allowing labels to be reused within sequence

    Parameters
    ----------
    sequence : Iterable[Hashable]
        The ordered sequence of labels to assign to nodes
        Labels can be reused as often as one likes without being
    label_attr : str, default 'label'
        The name of the attribute on each node to bind the label value to
    create_using, type[GraphLike], default Graph
        The type of networkx Graph to use to instantiate the path graph

    Returns
    -------
    sequence_graph : GraphLike
        The requisite graph, whose type is that of `create_using`
        Nodes in sequence_graph are the integer index of the label,
        while the labels themselves are bound to `label_attr` on each node
    """
    sequence_graph = path_graph(len(sequence), create_using=create_using)
    for i, label in enumerate(sequence):
        sequence_graph.nodes[i][label_attr] = label

    return sequence_graph


# TODO: handle index uniquification over union of path_graph_from_sequence


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
