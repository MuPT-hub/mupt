"""For converting back-and-forth between anytree and networkx DiGraphs"""

from typing import Any, Callable, Hashable, Iterable, Optional, TypeVar

from anytree.node import NodeMixin
from anytree.exporter import DictExporter
from anytree.iterators import AbstractIter, PreOrderIter

from networkx import DiGraph

from .subselect import NodePredicate

H = TypeVar("H", bound=Hashable)
type AttrIter = Iterable[tuple[Hashable, Any]]


def anytree_to_networkx(
    node: NodeMixin,
    dict_type: type = dict,
    iter_type: type[AbstractIter] = PreOrderIter,
    attr_iter: Optional[Callable[[AttrIter], AttrIter]] = None,
    filter_predicate: Optional[NodePredicate[NodeMixin]] = None,
    stop_predicate: Optional[NodePredicate[NodeMixin]] = None,
    max_depth: Optional[int] = None,
    node_converter: Callable[[NodeMixin], H] = lambda x: x,
) -> DiGraph:
    """Convert a tree into a directed NetworkX graph"""
    node_iter: AbstractIter = iter_type(
        node, filter_=filter_predicate, stop=stop_predicate, maxlevel=max_depth
    )
    node_exporter = DictExporter(
        dictcls=dict_type,
        attriter=attr_iter,
        childiter=list,  # tuple,
        # Will be exporting attributes on a node-by-node basis;
        # max level for tree traversal is set in node_iter.maxlevel
        maxlevel=1,
    )

    digraph = DiGraph()
    for node in node_iter:
        node_converted: H = node_converter(node)
        digraph.add_node(
            node_converted,
            **node_exporter.export(node),
        )
        if (parent := node.parent) is not None:
            parent_converted: H = node_converter(parent)
            if parent_converted in digraph:
                digraph.add_edge(parent_converted, node_converted)

    return digraph


def networkx_to_anytree(digraph: DiGraph) -> NodeMixin:
    """Convert a tree into a directed NetworkX graph"""
    raise NotImplementedError
