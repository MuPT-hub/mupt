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

from networkx.classes import Graph
from networkx.generators import path_graph
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
