"""
Monomer Interconnectivity and Degree (MID) graphs, for
encoding the topological connectivity of a polymer system
"""

from typing import (
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Optional,
)
from itertools import count
from functools import reduce

import networkx as nx
from networkx.classes import Graph


class TopologicalStructure(Graph):
    # DEV: opting not to call this just "Topology" for now to avoid confusion,
    # ...since many molecular packages also have a class by that name
    """
    An incidence topology induces on a set of Primitives,
    Represented as a Graph whose edge pairs generate the topology
    """

    ...


# graph generators
def path_graphs(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[Graph] = Graph,
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
    create_using: type[Graph] = Graph,
) -> nx.Graph:
    """
    Generate a single graph representing a collection of disjoint linear
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
