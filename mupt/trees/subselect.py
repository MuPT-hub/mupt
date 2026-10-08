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
    to_depth: Optional[int] = None,
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
    to_depth : Optional[int], default None
        An optional cap on the depth of the search;
        No nodes with depth to the root greater than
        or equal to `to_depth` will be yielded

    Yields
    ------
    node_selected : NodeMixin
        The first node along a given branch found to satisfy the predicate
    """
    # DEV: would be nice to support multiple roots to initialize search queue,
    # but that would require ensuring ALL pairs of roots are non-relatives,
    # which naively seems like an O(N^2) precheck (cleverer solutions probably exist)
    nodes_to_search: list[NodeLike] = [root]
    while nodes_to_search:
        curr_node = nodes_to_search.pop(0)
        if predicate(curr_node):
            yield curr_node

        # TB: +1 is annoying as hell, but is needed to be consistent with
        # anytree's `maxlevel` analog (I contend that max depth of 2 should
        # include nodes w/ depth 2, rather than bottoming out at 1; cest la vie)
        elif (to_depth is None) or ((curr_node.depth + 1) < to_depth - 1):
            # N.B.: with arbitrary successor function,
            # not guaranteed depth increases monotonically!
            nodes_to_search.extend(successors(curr_node))


pruned_BFS_subelection = primoprogenitors
