"""Convenience utilities for drawing Primitive hierarchies and connectivities"""

import logging

LOGGER = logging.getLogger(__name__)

from typing import Callable, Optional, Union

from networkx.classes import Graph, DiGraph
from matplotlib.axes import Axes

from .primitive import Primitive
from .simple.atomic import AtomicPrimitive
from ...chemistry.colors import COLORS_BY_ELEMENT
from ...graphs.visualization import draw_networkx_with_arcs, draw_networkx_tree
from ...trees.subselect import NodePredicate
from ...geometry.arraytypes import Vector2
from ...geometry.coordinates.reference import ORIGIN2


type Color = (
    str  # TODO: provide comprehensive typehint for all things mpl calls  colors
)
DEFAULT_COLOR: Color = "#1f78b4"  # same as network's default


def DEFAULT_COLORING_RULE(prim: "Primitive") -> Color:
    """
    Assign AtomicPrimitives an element-specific color, and
    all other Primitive a default color mathcing networkx
    """
    # TB: would prefer the simpler `is_atom(prim)` call, but need an isinstance
    # check to get linter to shut up about element not being a valid attribue
    if isinstance(prim, AtomicPrimitive):
        return COLORS_BY_ELEMENT.get(prim.element.symbol, DEFAULT_COLOR)
    else:
        return DEFAULT_COLOR


def visualize_cross_section(
    primitive: Primitive,
    cross_section: Union[Graph, NodePredicate["Primitive"]],
    base_arc_radius: float = 0.1,
    coloring_rule: Optional[Callable[["Primitive"], Color]] = DEFAULT_COLORING_RULE,
    use_shapes_for_pos: bool = True,
    default_pos: Vector2 = ORIGIN2,
    **kwargs,
) -> "Axes":
    """
    Draw a networkx Graph representation of the selected cross-section

    Can accept a pre-calculated cross-section or, if none is provided,
    will calculate the cross section on-the-spot before plotting

    See mupt.graphs.visualization.draw_networkx_with_arcs() for kwargs options
    """
    if not isinstance(cross_section, Graph):
        LOGGER.info(
            "Extracting cross-section from predicate, "
            "as no pre-computed cross-section was provided"
        )
        cross_section = primitive.cross_section(cross_section)

    if coloring_rule is not None:
        kwargs.setdefault("node_color", [coloring_rule(prim) for prim in cross_section])

    if use_shapes_for_pos:
        pos_from_shapes: dict[Primitive, Vector2] = {}
        for node_prim in cross_section:
            if node_prim.shape is None:
                center = default_pos.copy()
            else:
                ## DEV: 3D positions are rendered just fine by networkx,
                # but screw up draw_networkx_labels. Internally for render,
                # networkx just cuts off the z-coordinate, so we'll do the same here
                center = node_prim.shape.centroid[:-1]
            pos_from_shapes[node_prim] = center
        kwargs.setdefault("pos", pos_from_shapes)

    return draw_networkx_with_arcs(
        cross_section, base_arc_radius=base_arc_radius, **kwargs
    )


def visualize_hierarchy(
    primitive: Primitive,
    hierarchy_tree: Optional[DiGraph] = None,
    ax: Optional["Axes"] = None,
    **kwargs,
) -> "Axes":
    """
    Draw a networkx DiGraph representation of the partial order
    of this Primitive and all Primitives below it in the hierarchy

    Can accept a pre-calculated hierarchy tree or, if None is provided,
    will extract one from the current Primitive

    See mupt.graphs.visualization.draw_networkx_tree() for kwargs options
    """
    if hierarchy_tree is None:
        hierarchy_tree = primitive.hierarchy_tree()

    return draw_networkx_tree(hierarchy_tree, ax=ax, **kwargs)
