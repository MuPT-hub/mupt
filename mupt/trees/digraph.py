"""For converting back-and-forth between anytree and networkx DiGraphs"""

from typing import Any, Callable, Hashable, Iterable, Optional, Type

from anytree.node import NodeMixin
from anytree.exporter import DictExporter
from anytree.iterators import AbstractIter, PreOrderIter

from networkx import DiGraph

from .subselect import NodePredicate

type AttrIter = Iterable[tuple[Hashable, Any]]


def anytree_to_networkx(
    node: NodeMixin,
    dict_type: Type = dict,
    iter_type: Type[AbstractIter] = PreOrderIter,
    attr_iter: Optional[Callable[[AttrIter], AttrIter]] = None,
    filter_predicate: Optional[NodePredicate[NodeMixin]] = None,
    stop_predicate: Optional[NodePredicate[NodeMixin]] = None,
    max_depth: Optional[int] = None,
    node_converter: Callable[[NodeMixin], Hashable] = lambda x: x,
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
    for subnode in node_iter:
        digraph.add_node(
            node_converter(subnode),
            **node_exporter.export(subnode),
        )
        if (parent := subnode.parent) is not None:
            digraph.add_edge(parent, subnode)

    return digraph


def networkx_to_anytree(digraph: DiGraph) -> NodeMixin:
    """Convert a tree into a directed NetworkX graph"""
    raise NotImplementedError
