"""Utilities for drawing graphs, including multigraphs with parallel edges"""

from typing import Generator, Optional, TYPE_CHECKING

from inspect import signature
from collections import defaultdict

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

from networkx import Graph, MultiGraph
from networkx.drawing import (
    spring_layout,
    draw_networkx,
    draw_networkx_edges,
)

from .graphtypes import GraphEdge, GraphPositions


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

        arc_radii: Generator[float, None, None] = (
            base_arc_radius * increment
            for increment in range(1 - num_parallel_edges, num_parallel_edges, 2)
        )
        for arc_radius, edge in zip(arc_radii, parallel_edges):
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

    if "with_labels" not in kwargs:
        kwargs["with_labels"] = True

    edgelist = kwargs.pop("edgelist", G.edges)  # handle edgelist manually
    _ = kwargs.pop("connectionstyle", None)  # prevent connectionstyle override

    # TB DEV: these kwargs filters are lifted from internals of draw_networkx()
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
            arrows=True,  # suppress warning on edge w/ connectionstyle on simple Graphs
            **edge_kwargs,
        )

    return ax
