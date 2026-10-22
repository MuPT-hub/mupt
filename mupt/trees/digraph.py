"""For converting back-and-forth between anytree and networkx DiGraphs"""

from typing import (
    Callable,
    Hashable,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

from anytree.node import NodeMixin
from anytree.exporter import DictExporter
from anytree.iterators import AbstractIter, PreOrderIter

from networkx import DiGraph

from .subselect import NodePredicate

H = TypeVar("H", bound=Hashable)


def anytree_to_networkx(
    node: NodeMixin,
    node_exporter: Optional[DictExporter] = None,
    iter_type: type[AbstractIter] = PreOrderIter,
    filter_predicate: Optional[NodePredicate[NodeMixin]] = None,
    stop_predicate: Optional[NodePredicate[NodeMixin]] = None,
    max_depth: Optional[int] = None,
    node_converter: Union[
        Callable[[NodeMixin], H],
        Mapping[NodeMixin, H],
    ] = lambda x: x,
) -> DiGraph:
    """Convert a tree into a directed NetworkX graph"""
    if isinstance(node_converter, Mapping):

        def _node_converter(nd: NodeMixin) -> H:
            return node_converter[nd]

    else:
        _node_converter = node_converter  # type: ignore

    # dict_exporter.maxlevel is forced 1 to avoid copying nodes as attributes
    # not to be confused with max level for tree traversal)
    if node_exporter is None:
        node_exporter = DictExporter()
    node_exporter.maxlevel = 1

    node_iter: AbstractIter = iter_type(
        node, filter_=filter_predicate, stop=stop_predicate, maxlevel=max_depth
    )

    digraph = DiGraph()
    for subnode in node_iter:
        node_converted: H = _node_converter(subnode)
        digraph.add_node(
            node_converted,
            **node_exporter.export(subnode),
        )
        if (parent := subnode.parent) is not None:
            parent_converted: H = _node_converter(parent)
            if parent_converted in digraph:
                digraph.add_edge(parent_converted, node_converted)

    return digraph


def networkx_to_anytree(digraph: DiGraph) -> NodeMixin:
    """Convert a tree into a directed NetworkX graph"""
    raise NotImplementedError
