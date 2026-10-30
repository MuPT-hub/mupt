"""For converting back-and-forth between anytree and networkx DiGraphs"""

from typing import (
    Callable,
    Hashable,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

from anytree.exporter import DictExporter
from anytree.iterators import PreOrderIter

from networkx import DiGraph

from .subselect import NodeLike

H = TypeVar("H", bound=Hashable)


# TB: supressing linter complexity (C901) warning for now,
# but in the future this should be refactored to be more modular
# and contain less branched business logic in one place
def anytree_to_networkx(  # noqa: C901
    node: NodeLike,
    node_exporter: Optional[DictExporter] = None,
    max_depth: Optional[int] = None,
    node_converter: Union[Callable[[NodeLike], H], Mapping[NodeLike, H]] = lambda x: x,
) -> DiGraph:
    """
    Convert a tree represented by an anytree NodeMixin to a directed NetworkX graph

    Parameters
    ----------
    node : NodeMixin
        The root node to begin tree traversal from

        This can be any object which inherits from
        the anytree NodeMixin, including anytree Node
    node_exporter: Optional[DictExporter] = None,
        An optional DictExporter instance which prescribes how node
        attr data should be mapped into attributes on networkx Graph nodes

        N.B.: any DictExporter instance passed in here MUST have maxlevel=1
        this ensures other nodes don't get duplicated as Graph node attributes
    max_depth: Optional[int] = None,
        An optional number of iterations from the root to search before halting

        If None, no such search depth limit is enforced
    node_converter: Union[Callable[[NodeMixin], H], Mapping[NodeMixin, H]]
            default lambda x: x
        An optional mechanism for converting node objects to other
        objects for use as nodes in the final networkx DiGraph
        Can be either a callable object like a function,
        or a mapping keyed by anytree NodeMixin objects like a dict

        If left unspecified (i.e. None), makes networkx nodes
        the literal NodeMixin objects in the original tree

    Returns
    -------
    digraph : DiGraph
        A directed networkx graph representing the (filtered) tree

        N.B.: the returned graph MAY have multiple connected components;
        this can happen, for instance, if all intermediate nodes are filtered out
    """
    if isinstance(node_converter, Mapping):
        node_converter = node_converter.__getitem__

    if node_exporter is None:
        # force dict_exporter.maxlevel=1 to avoid recursively copying other nodes``
        # as attributes; not to be confused with max level for tree traversal)
        node_exporter = DictExporter(maxlevel=1)

    if node_exporter.maxlevel != 1:
        raise ValueError("Externally-supplied DictExporter must have maxlevel=1")

    # determine which nodes survive and who their nearest ancestors are
    node_iter = PreOrderIter(node, maxlevel=max_depth)

    predecessors: dict[NodeLike, NodeLike] = {}
    survivors: dict[NodeLike, H] = dict()
    for subnode in node_iter:  # iter excludes certain nodes
        survivors[subnode] = node_converter(subnode)
        if (parent := subnode.parent) is not None:
            predecessors[subnode] = parent

    ancestors_surviving: dict[NodeLike, NodeLike] = dict()
    for survivor_with_parent in set(predecessors):
        # not using dict.items() to avoid modifying in-place if no ancestor can be found
        last_seen_ancestor: NodeLike = predecessors[survivor_with_parent]
        if last_seen_ancestor in survivors:
            continue

        if last_seen_ancestor in ancestors_surviving:
            predecessors[survivor_with_parent] = ancestors_surviving[last_seen_ancestor]
        else:
            # TB: a potential optimization would be to use ancestors_surviving to skip
            # over parts of the path for which we know the nearest surviving ancestor.
            # Omitting for now, since the tree depths we expect to encounter likely
            # never will be so large that the benefits offset the extra complexity
            for grand_ancestor in last_seen_ancestor.ancestors:
                if grand_ancestor in survivors:
                    predecessors[survivor_with_parent] = grand_ancestor
            else:
                # no surviving relatives found
                del predecessors[survivor_with_parent]

    # assemble graph, with necessary conversions
    digraph = DiGraph()
    for survivor, survivor_converted in survivors.items():
        # N.B.: must add ALL nodes before creating edges, to ensure no conversion
        # or attribute transfer is skipped due to a particular traversal order
        digraph.add_node(survivor_converted, **node_exporter.export(survivor))

    for survivor, predecessor in predecessors.items():
        digraph.add_edge(survivors[predecessor], survivors[survivor])

    return digraph


# def networkx_to_anytree(digraph: DiGraph) -> NodeMixin:
#     """Convert a tree into a directed NetworkX graph"""
#     raise NotImplementedError
