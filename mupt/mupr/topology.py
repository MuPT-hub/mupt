"""
Utilities for graphs which encode neighbor connectivity
of chemical systems in the MuPT molecular representation
"""

from typing import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    Union,
    TYPE_CHECKING,
)
from inspect import signature
from itertools import count
from functools import reduce
from collections import Counter, defaultdict

from numpy import ndarray, arange
from networkx.classes import Graph, MultiGraph
from networkx.generators import path_graph
from networkx.algorithms import union as graph_union
from networkx.drawing import (
    spring_layout,
    draw_networkx,
    draw_networkx_edges,
)

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

Node = Hashable
Edge = tuple[Node, Node]
MultiEdge = tuple[Node, Node, int]
GraphEdge = Union[Edge, MultiEdge]

type GraphPositions = dict[Node, ndarray]
type GraphLayout = Callable[[Graph], GraphPositions]


# Network properties
def is_indiscrete(graph: Graph) -> bool:
    """Whether the current topology represents an indiscrete topology
    i.e. a "trivial topology" without any connections
    """
    return graph.number_of_edges() == 0


is_trivial = is_indiscrete


def is_empty(graph: Graph) -> bool:
    """
    Whether the topology is empty (i.e. has no nodes)

    Represents a valid topology, since it contains the empty set
    and itself (which just so happens to also be the empty set)
    """
    return graph.number_of_nodes() == 0


def is_unbranched(graph: Graph) -> bool:
    """Whether the topology contains only unbranching chain(s) or isolated nodes"""
    return all(node_deg <= 2 for node_id, node_deg in graph.degree)


is_linear = is_unbranched


def is_branched(graph: Graph) -> bool:
    """Whether the topology contains any branching nodes"""
    return not is_unbranched(graph)


def termini(graph: Graph) -> Generator[int, None, None]:
    """
    Generates the indices of all nodes corresponding to terminal primitives
    (i.e. those with only one outgoing bond)
    """
    for node_idx, degree in graph.degree:
        if degree == 1:
            yield node_idx


leaves = termini


def canonical_graph_property(graph: Graph) -> str:
    """
    Return a canonical form based on the graph structure and coloring
    induced by the canonical forms of internal Primitives
    Tantamount to solving the graph isomorphism problem
    """
    # raise NotImplementedError('Graph canonicalization is not implemented yet')
    # return nx.weisfeiler_lehman_graph_hash(self)
    # # stand-in for more specific implementation to follow

    return str(
        hash(
            # temporary, quick-to-compute stand-in
            # for eventual "real-deal" canonical form
            tuple(Counter(deg for node, deg in graph.degree).items())
        )
    )


# graph generators
def path_graphs(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[Graph] = Graph,
) -> Generator[Graph, None, None]:
    """
    Generate a sequence of path graphs according to a
    provided sequence of lengths and labelling scheme
    """
    if node_labels is None:
        node_labels = count(start=0, step=1)

    for chain_length in chain_lengths:
        yield path_graph(
            (next(node_labels) for _ in range(chain_length)),
            create_using=create_using,
        )


def noodle_graph(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[Graph] = Graph,
) -> Graph:
    """
    Generate a single topology representing a collection of disjoint linear
    chains according to a provided sequence of lengths and labelling scheme
    """
    return reduce(
        graph_union,
        path_graphs(
            chain_lengths=chain_lengths,
            node_labels=node_labels,
            create_using=create_using,
        ),
    )


# visualisation
def determine_arc_radii(
    graph: Graph | MultiGraph,
    base_arc_radius: int | float = 0.1,
) -> dict[GraphEdge, str]:
    """
    Configure a graph to display symmetric-looking arcs for
    (possibly parallel) edges rendered when drawing that graph

    Arc radii will be scaled up from the chosen base radius, e.g.
    * N=1: O----O ==> (0,)
            ____
    * N=2: O    O ==> (-1, 1)
            ▔▔
            ____
    * N=3: O----O ==> (-1, 0, 1)
            ▔▔
    etc.
    """
    # truncate to second element to cut off edge key in the case of MultiGraph
    # only want to compare on the basis of the unordered pair of nodes
    edges_by_node_pair = defaultdict(list)
    for edge in graph.edges:
        a, b, *_ = edge
        edges_by_node_pair[frozenset((a, b))].append(edge)

    conn_style_map: dict[GraphEdge, str] = dict()
    for parallel_edges in edges_by_node_pair.values():
        num_parallel_edges: int = len(parallel_edges)

        for arc_radius, edge in zip(
            base_arc_radius * arange(1 - num_parallel_edges, num_parallel_edges, 2),
            parallel_edges,
        ):
            conn_style_map[edge] = f"arc3,rad={arc_radius}"

    return conn_style_map


def draw_networkx_with_arcs(
    G: Graph | MultiGraph,  # TB: MultiGraph < Graph already; just making explicit
    pos: Optional[GraphPositions] = None,
    ax: Optional["Axes"] = None,
    base_arc_radius: float = 0.1,
    default_margins: float = 0.25,
    **kwargs,
) -> "Axes":
    """
    Draw the graph G with separated arcs for parallel edges

    Thin wrapper around draw_networkx() and draw_networkx_edges()
    See networkx.drawing documentation for kwarg details
    """
    from matplotlib.pyplot import figure

    # TB DEV: these kwargs filters are lifted from internals of draw_networkx()
    kwargs["with_labels"] = True
    edgelist = kwargs.pop("edgelist", G.edges)  # handle edgelist manually
    _ = kwargs.pop("connectionstyle", None)  # prevent connectionstyle override

    valid_edge_kwds = signature(draw_networkx_edges).parameters.keys()
    edge_kwargs = {k: v for k, v in kwargs.items() if k in valid_edge_kwds}

    if ax is None:
        fig = figure()
        ax = fig.add_axes((0, 0, 1, 1))
        ax.margins(default_margins)
        ax.set_axis_off()

    if pos is None:
        pos = spring_layout(G)  # default to spring layout

    # N.B.: the "edge_indices" logic inside draw_networkx() doesn't correctly handle
    # distinct connectionstyles for each parallel multiedge in a multigraph
    # viz. -|#|= edges in 4-node path unexpectedly refs styles like [0 | 0 1 2 | 0 1])
    # Drawing edge-by-edge ensures the arcs drawn respect their assigned connectionstyle
    draw_networkx(G, pos=pos, ax=ax, edgelist=[], **kwargs)  # skip edge drawing here
    arc_styles = determine_arc_radii(G, base_arc_radius=base_arc_radius)
    for edge in edgelist:
        draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            edgelist=[edge],
            connectionstyle=arc_styles[edge],
            **edge_kwargs,
        )

    return ax
