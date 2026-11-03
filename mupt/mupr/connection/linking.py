"""
Utilities for linking Connectors to form two-way bonded connections
in a MuPT representation based on global topology specification
"""

import logging

LOGGER = logging.getLogger(__name__)

from typing import (
    Collection,
    Mapping,
    TypeVar,
)

T = TypeVar("T")

from networkx import Graph

from .connectors import Connector


class BijectionError(ValueError):
    """
    Raised when a pair of objects expected to
    be in 1-to-1 correspondence are mismatched
    """

    ...


class GraphLinkingError(ValueError):
    """Raised when an invalid mapping to a graph is encountered"""

    ...


class NodeMappingError(GraphLinkingError):
    """
    Raised when an invalid mapping between an
    object and a graph node is encountered
    """

    ...


class EdgeMissingError(GraphLinkingError):
    """
    Raised when an invalid mapping between a
    pair of objects and a graph edge is encountered
    """

    ...


def DEFAULT_ITER_RULE(graph: Graph) -> int:
    """
    A default GraphIterRule implementation, to be
    used when no other specific rule is supplied
    """
    # TB DEV: 10 is just a generaous scaling I made up :P - benchmarks needed
    return 10 * graph.number_of_nodes()


def _check_connectors_cover_topology(
    # TB: if Graph supported Generic subscripting, this
    # annotation would be Graph[T], indicating node type
    topology: Graph,
    # Collection (rather than Iterable) needed for length check
    mapped_connectors: Mapping[T, Collection[Connector]],
) -> None:
    """
    Necessary (but not sufficient) conditions to ensure a map from
    graph nodes to collections of Connectors covers all nodes and edges

    Specifically, checks that:
    * The preimage of map contains node set
      I.e. every node gets some collection of Connectors
    * The image of each node has no fewer Connectors than the node has neighbors

    Returns silently if passing; raises NodeMappingError otherwise
    """
    if not set(topology.nodes).issubset(set(mapped_connectors.keys())):
        # Weaker size requirement; nodes need not be in 1:1 correspondence
        # with Connector collections, merely covered by them
        raise NodeMappingError(
            "Not all nodes in the requested topology "
            "are covered by collections of Connectors"
        )

    for node in topology.nodes:
        if (num_connectors := len(mapped_connectors[node])) < (
            num_neighbors := topology.degree[node]
        ):
            raise NodeMappingError(
                f"Node {node!r} has {num_neighbors} neighbors, but only "
                f"{num_connectors} connection to distribute among them"
            )
