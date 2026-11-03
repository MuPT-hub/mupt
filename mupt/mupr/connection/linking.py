"""
Utilities for linking Connectors to form two-way bonded connections
in a MuPT representation based on global topology specification
"""


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
