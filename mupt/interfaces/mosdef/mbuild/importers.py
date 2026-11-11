"""
Read coordinates from an mBuild Compound back onto an existing MuPT Primitive hierarchy.

The Compound must come from `primitive_to_mbuild`, which tags every Compound it
creates with the `address` of the Primitive it was made from. Positions are
matched to Primitives by that address, so the Compound's particle order and
naming do not matter.
"""

from typing import Optional

import numpy as np
from anytree.iterators import PostOrderIter, PreOrderIter
from mbuild.compound import Compound
from scipy.spatial.transform import RigidTransform

from ....geometry.shapes import PointCloud
from ....mupr.connection.connectors import Connector
from ....mupr.primitives.primitive import Primitive, PrimitiveAddress


def _connectors_at_or_below(prim: Primitive) -> list[Connector]:
    """Every distinct Connector held by a Primitive or any of its descendants"""
    seen: dict[int, Connector] = dict()
    for node in PreOrderIter(prim):
        for connector in node.connections.connectors:
            seen.setdefault(id(connector), connector)

    return list(seen.values())


def _translate_primitive(prim: Primitive, new_centroid: np.ndarray) -> None:
    """
    Move a Primitive (shape, descendant shapes and Connectors) so that its
    centroid lands on new_centroid. A Primitive with no shape is given a
    single-point PointCloud at new_centroid and its Connectors' anchors
    are moved onto that point.
    """
    if prim.shape is None:
        prim.shape = PointCloud(positions=new_centroid)
        for connector in prim.connections.connectors:
            offset = new_centroid - connector.anchor.position
            connector.rigidly_transform(RigidTransform.from_translation(offset))
        return

    offset = new_centroid - np.asarray(prim.shape.centroid, dtype=float)
    translation = RigidTransform.from_translation(offset)
    for node in PreOrderIter(prim):
        node._rigidly_transform_shape(translation)
    for connector in _connectors_at_or_below(prim):
        connector.rigidly_transform(translation)


def _reaim_bonded_connectors(moved: list[Primitive]) -> None:
    """Point each bonded Connector's linker at its neighbor's (updated) anchor"""
    for prim in moved:
        for connector in _connectors_at_or_below(prim):
            if connector.has_neighbor:
                connector.linker.position = np.array(
                    connector.neighbor.anchor.position, dtype=float
                )
                connector.neighbor.linker.position = np.array(
                    connector.anchor.position, dtype=float
                )


def _refresh_ancestor_shapes(
    root: Primitive,
    moved: list[Primitive],
    old_centroids: dict[PrimitiveAddress, np.ndarray],
) -> None:
    """
    Bring the shapes of Primitives above the moved ones back in line with their
    children: PointClouds are rebuilt from their children's points, other shapes
    are translated by the shift in their children's mean centroid, and Primitives
    without a shape are left without one.
    """
    moved_ids = {id(prim) for prim in moved}
    stale: set[int] = set()
    for prim in moved:
        for ancestor in prim.iter_path_reverse():
            if ancestor is prim:
                continue
            stale.add(id(ancestor))
            if ancestor is root:
                break

    for node in PostOrderIter(root):  # children before parents
        if (id(node) not in stale) or (id(node) in moved_ids) or (node.shape is None):
            continue

        child_points = _child_points(node)
        if child_points is None:
            continue

        if isinstance(node.shape, PointCloud):
            node.shape = PointCloud(positions=child_points)
        elif node.address in old_centroids:
            offset = child_points.mean(axis=0) - old_centroids[node.address]
            node._rigidly_transform_shape(RigidTransform.from_translation(offset))


def _child_points(prim: Primitive) -> Optional[np.ndarray]:
    """
    Stack the points of a Primitive's children as an (N, 3) array: every point of
    a PointCloud child, the centroid of any other shaped child. None if no child
    has a shape.
    """
    points = [
        (
            np.asarray(child.shape.positions, dtype=float).reshape(-1, 3)
            if isinstance(child.shape, PointCloud)
            else np.asarray(child.shape.centroid, dtype=float).reshape(1, 3)
        )
        for child in prim.children
        if child.shape is not None
    ]
    return np.vstack(points) if points else None


def update_primitive_from_mbuild(
    prim: Primitive,
    compound: Compound,
    nm_to_coords: float = 10.0,
) -> list[Primitive]:
    """Write the positions of an mBuild Compound back onto an existing MuPT hierarchy.

    Each particle (leaf Compound) of `compound` is matched to the Primitive in
    `prim`'s hierarchy that has the same address, which `primitive_to_mbuild`
    stores on every Compound it creates. The matched Primitive is translated so
    that its centroid sits at the particle's position: its shape, the shapes of
    any Primitives below it, and its Connectors all move with it. Afterwards,
    bonded Connectors are re-aimed at their neighbors, and the shapes of the
    Primitives above the moved ones are refreshed (PointClouds rebuilt from
    their children, other shapes translated, absent shapes left absent).

    Typical use: export with `primitive_to_mbuild`, relax or simulate the
    Compound's coordinates elsewhere, set them on that same Compound (e.g.
    `compound.xyz = new_positions_nm`), then call this function.

    Parameters
    ----------
    prim : Primitive
        The Primitive that was passed to `primitive_to_mbuild` (or any ancestor of it).
        It is updated in place.
    compound : mbuild.Compound
        A Compound produced by `primitive_to_mbuild`, with updated particle positions.
        Coordinates must already be unwrapped across periodic boundaries.
    nm_to_coords : float, default 10.0
        Factor converting mBuild's nanometers into the hierarchy's length unit.
        The default (10.0) converts to Angstrom; use the reciprocal of the
        `coords_to_nm` passed to `primitive_to_mbuild`.

    Returns
    -------
    list[Primitive]
        The Primitives whose positions were set, in the Compound's particle order.

    Raises
    ------
    ValueError
        If a particle carries no MuPT address, its address is not found in
        `prim`'s hierarchy, or its position is not finite.
    """
    by_address: dict[PrimitiveAddress, Primitive] = {
        node.address: node for node in PreOrderIter(prim)
    }

    targets: list[tuple[Primitive, np.ndarray]] = []
    for particle in compound.particles():
        address = getattr(particle, "address", None)
        if address is None:
            raise ValueError(
                f"mBuild particle {particle.name!r} has no MuPT address; "
                "only Compounds created by primitive_to_mbuild can be read back"
            )
        if address not in by_address:
            raise ValueError(
                f"mBuild particle {particle.name!r} refers to Primitive address "
                f"{address!r}, which is not in the hierarchy of {prim}"
            )
        position = np.asarray(particle.pos, dtype=float) * nm_to_coords
        if not np.all(np.isfinite(position)):
            raise ValueError(
                f"mBuild particle {particle.name!r} has a non-finite "
                f"position {position}"
            )
        targets.append((by_address[address], position))

    old_centroids: dict[PrimitiveAddress, np.ndarray] = dict()
    for node in PreOrderIter(prim):
        if (node.shape is not None) and not isinstance(node.shape, PointCloud):
            child_points = _child_points(node)
            if child_points is not None:
                old_centroids[node.address] = child_points.mean(axis=0)

    moved = [target for target, _ in targets]
    for target, position in targets:
        _translate_primitive(target, position)
    _reaim_bonded_connectors(moved)
    _refresh_ancestor_shapes(prim, moved, old_centroids)

    return moved
