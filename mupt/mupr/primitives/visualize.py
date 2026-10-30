"""Convenience utilities for drawing Primitive hierarchies and connectivities"""

from typing import Optional, Union

from networkx.classes import DiGraph
from matplotlib.axes import Axes

from .primitives_legacy import PrimitiveLegacy, PrimitiveLabel, PrimitiveHandle
from ...graphs.visualization import draw_networkx_with_arcs, draw_networkx_tree
from ...geometry.arraytypes import Vector2
from ...geometry.coordinates.reference import ORIGIN2


def visualize_topology(
    prim: PrimitiveLegacy,
    ax: Optional["Axes"] = None,
    base_arc_radius: float = 0.1,
    use_shapes_for_pos: bool = True,
    default_pos: Vector2 = ORIGIN2,
    # TODO: add support for coloring rules
    **draw_kwargs,
) -> "Axes":
    """
    Draw the connectivity of a Primitive's children to the passed Axes

    See mupt.graphs.visualization.draw_networkx_with_arcs() for kwargs options
    """
    graph = prim.topology
    if use_shapes_for_pos:
        pos_from_shapes: dict[PrimitiveLegacy, Vector2] = {}
        for child_handle in graph.nodes:
            child = prim.fetch_child(child_handle)
            if child.shape is None:
                center = default_pos.copy()
            else:
                ## DEV: 3D positions are rendered just fine by networkx,
                # but screw up draw_networkx_labels. Internally for render,
                # networkx just cuts off the z-coordinate, so we'll do the same here
                center = child.shape.centroid[:-1]

            pos_from_shapes[child_handle] = center
        draw_kwargs["pos"] = pos_from_shapes

    return draw_networkx_with_arcs(
        graph,
        ax=ax,
        base_arc_radius=base_arc_radius,
        **draw_kwargs,
    )


def visualize_hierarchy(
    prim: PrimitiveLegacy,
    hierarchy_tree: Optional[DiGraph] = None,
    use_handles_for_labels: bool = True,
    root_label: Optional[PrimitiveLabel] = None,
    **kwargs,
) -> "Axes":
    """
    Draw a networkx DiGraph representation of the partial order
    of a Primitive and all Primitives below it in the hierarchy

    Can accept a pre-calculated hierarchy tree or, if None is provided,
    will extract one from the current Primitive

    See mupt.graphs.visualization.draw_networkx_tree() for kwargs options
    """
    if hierarchy_tree is None:
        hierarchy_tree = prim.hierarchy_tree()

    if use_handles_for_labels:
        label_map: dict[PrimitiveLegacy, Union[PrimitiveHandle, PrimitiveLabel]] = {
            subprim: handle
            for prim in hierarchy_tree
            for handle, subprim in prim.children_by_handle.items()
        }
        label_map[prim] = prim.label if (root_label is None) else root_label

        kwargs["with_labels"] = True
        kwargs["labels"] = label_map
        print(label_map)

    return draw_networkx_tree(hierarchy_tree, **kwargs)
