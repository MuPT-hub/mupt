"""Search and selection routines over tree nodes"""

from typing import (
    Callable,
    Generator,
    Iterable,
    Optional,
    TypeVar,
    TypeAlias,
)
from anytree.node import NodeMixin

NodeLike = TypeVar("NodeLike", bound=NodeMixin, covariant=True)
NodePredicate: TypeAlias = Callable[[NodeLike], bool]


# TB DEV: would be nice to eventually integrate w/ anytree's AbstractIter;
# Behavior of 'stop' criterion is different enough that I've kept separate for now
def primoprogenitors(
    root: NodeLike,
    predicate: NodePredicate[NodeLike],
    maxlevel: Optional[int] = None,
    successors: Callable[[NodeLike], Iterable[NodeLike]] = lambda node: node.children,
) -> Generator[NodeLike, None, None]:
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
    predicate: NodeNodePredicate
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
    # DEV: would be nice to support multiple roots to initialize,
    # but would require ensuring no root is relative of any other
    # root in list, which naively seems like an O(N^2) check
    nodes_to_search: list[NodeLike] = [root]
    while nodes_to_search:
        curr_node = nodes_to_search.pop(0)
        if predicate(curr_node):
            yield curr_node
        elif (maxlevel is None) or (curr_node.depth <= maxlevel):
            nodes_to_search.extend(successors(curr_node))


pruned_BFS_subelection = primoprogenitors
