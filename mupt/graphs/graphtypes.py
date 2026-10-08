"""Typehints useful when writing graph-related code"""

from typing import Callable, Hashable, Mapping, TypeVar, Union

from numpy import ndarray
from networkx import Graph, DiGraph, MultiGraph  # noqa: F401


Node = Hashable  # N.B.: not the same as anytree.Node! (consider disambiguating)
Edge = tuple[Node, Node]
MultiEdge = tuple[Node, Node, int]
GraphEdge = Union[Edge, MultiEdge]

GraphLike = TypeVar("GraphLike", bound=Graph, contravariant=True)

type GraphPositions = Mapping[Node, Union[ndarray, tuple[float, ...]]]
type GraphLayout = Callable[[Graph], GraphPositions]
