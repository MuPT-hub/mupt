"""Utilities for drawing graphs"""

from typing import Generator, Optional, TYPE_CHECKING
from collections import defaultdict

from networkx.classes import Graph, DiGraph, MultiGraph
from networkx.drawing import kamada_kawai_layout, draw as draw_nx, draw_networkx

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from .graphtypes import GraphEdge, GraphLayout, GraphPositions
from .invariants import is_indiscrete


def default_graph_axes(margins: float = 0.25) -> "Axes":
    """
    Default strategy for creating a blank single-axis figure to draw graphs onto

    Useful as default for creating axes when None have been provided
    """
    from matplotlib.pyplot import figure

    fig = figure()
    ax = fig.add_axes((0, 0, 1, 1))  # full figure
    ax.margins(margins)
    ax.set_axis_off()

    return ax


def graph_repr(graph: Graph) -> str:
    """
    Temporary port of deprecated TopologicalStructure.__repr__
    Moved over for backwards compatibility until Primitive is reworked
    """
    return (
        f"{graph.__class__.__name__}"
        f"(num_objects={graph.number_of_nodes()}, "
        f"indiscrete={is_indiscrete(graph)})"
    )


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


def draw_graph(
    graph: Graph,
    ax: Optional["Axes"] = None,
    layout: GraphLayout = kamada_kawai_layout,
    **draw_kwargs,
) -> None:
    """Draw the topology's graph"""
    if "with_labels" not in draw_kwargs:
        draw_kwargs["with_labels"] = True

    draw_nx(
        graph,
        ax=ax,
        pos=layout(graph),
        **draw_kwargs,
    )


def draw_networkx_tree(
    tree: DiGraph,
    pos: Optional[GraphPositions] = None,
    ax: Optional["Axes"] = None,
    margins: float = 0.25,
    **kwargs,
) -> "Axes":
    """
    Draw a directed graph with separated arcs for parallel edges

    Thin wrapper around draw_networkx()
    See networkx.drawing documentation for kwarg details
    """
    from networkx import nx_agraph  # nice pygraphviz layouts

    if "with_labels" not in kwargs:
        kwargs["with_labels"] = True

    if ax is None:
        ax = default_graph_axes(margins=margins)

    if pos is None:
        pos = nx_agraph.graphviz_layout(tree, prog="dot")

    draw_networkx(tree, pos=pos, **kwargs)

    return ax
