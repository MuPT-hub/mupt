"""Utilities for drawing graphs"""

from typing import Callable, Hashable, Optional, TYPE_CHECKING

from networkx.classes import Graph
from networkx.drawing import kamada_kawai_layout, draw as draw_nx

from numpy import ndarray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from .invariants import is_indiscrete

type GraphLayout = Callable[[Graph], dict[Hashable, ndarray]]


def graph_repr(graph: Graph) -> str:
    """
    Temporary port of deprecated TopologicalStructure.__repr__
    Moved over
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
