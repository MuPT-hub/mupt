"""Search and selection routines over trees nodes"""

from typing import Callable, Generator, Optional
from anytree.node import NodeMixin

type NodePredicate = Callable[[NodeMixin], bool]


def primoprogenitors(
    root: NodeMixin,
    predicate: NodePredicate,
    maxlevel: Optional[int] = None,
) -> Generator[NodeMixin, None, None]:
    """
    Subselect the first node along each branch of a tree which
    satisfies the given predicate. Nodes yielded have the property
    that NONE of their ancestors satisfy the predicate

    Searches the tree in BFS order with pruning,
    i.e. does not search below selected node

    Parameters
    ----------
    root: NodeMixin
        The node highest in the tree to be traversed
    predicate: NodePredicate
        A callable indicating whether a node should be selected
    maxlevel : Optional[int], default None
        An optional cap on the depth of the search;
        No nodes with depth to the root greater than
        `maxlevel` will be yielded

    Yields
    ------
    node_selected : NodeMixin
        The first node along a given branch found to satisfy the predicate
    """
    nodes_to_search: list[NodeMixin] = [root]
    while nodes_to_search:
        curr_node = nodes_to_search.pop(0)
        if predicate(curr_node):
            yield curr_node
        elif (maxlevel is None) or (curr_node.depth <= maxlevel):
            nodes_to_search.extend(curr_node.children)


pruned_BFS_subelection = primoprogenitors
