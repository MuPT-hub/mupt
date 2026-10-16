"""Utilities for drawing graphs"""

from typing import Optional, TYPE_CHECKING

from networkx.classes import Graph
from networkx.drawing import kamada_kawai_layout, draw as draw_nx

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from .graphtypes import GraphLayout
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
