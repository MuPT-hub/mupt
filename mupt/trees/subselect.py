"""Search and selection routines over tree nodes"""

from typing import (
    Callable,
    Generator,
    Iterable,
    Optional,
    TypeVar,
    TypeAlias,
)
from collections import deque

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
    predicate: NodePredicate
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
    # DEV: could potentially support multiple initial roots, provided one has an
    # efficient way to precheck ALL pairs of those roots are not ancestors/descendants
    ROOT_DEPTH: int = 0
    if (to_depth is not None) and (to_depth <= ROOT_DEPTH):
        return

    nodes_to_search: deque[tuple[NodeLike, int]] = deque([(root, ROOT_DEPTH)])
    while nodes_to_search:
        node, depth = nodes_to_search.popleft()  # O(1) pop which preserves BFS order
        if predicate(node):
            yield node

        # TB: +1 is annoying as hell, but is needed to be consistent with
        # anytree's `maxlevel` analog (I contend that max depth of 2 should
        # include nodes w/ depth 2, rather than bottoming out at 1; cest la vie)
        elif (to_depth is None) or ((depth + 1) < to_depth):
            nodes_to_search.extend(
                (successor, depth + 1) for successor in successors(node)
            )
            # N.B.: with arbitrary successor function, node.depth is not guaranteed to
            # increase monotonically, nevermind incrementally; local tally combats this


pruned_BFS_subelection = primoprogenitors
