"""Typehints useful when writing graph-related code"""

from typing import Callable, Hashable, Union

from numpy import ndarray
from networkx import Graph, DiGraph, MultiGraph  # noqa: F401


Node = Hashable
Edge = tuple[Node, Node]
MultiEdge = tuple[Node, Node, int]
GraphEdge = Union[Edge, MultiEdge]

type GraphPositions = dict[Node, ndarray]
type GraphLayout = Callable[[Graph], GraphPositions]
