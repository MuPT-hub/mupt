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
from ....mupr.primitives.primitive import Primitive, PrimitiveAddress
from ..._shared.positions import BeadPosition, bead_position_of, check_bead_position


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


def _refresh_ancestor_shapes(root: Primitive, moved: list[Primitive]) -> None:
    """
    Rebuild the shapes of the Primitives between the moved ones and `root`
    (inclusive) as PointClouds of their children's points, children first.
    Primitives without a shape are left without one.

    DEV: this assumes that, when interfacing with mBuild, every shaped Primitive
    above the exported cross-section is a PointCloud (as the RDKit importers
    build them). Any other shape on such a Primitive (e.g. a Sphere) is replaced
    by a PointCloud here. Revisit if non-PointCloud parents need to be preserved.
    """
    moved_ids = {id(prim) for prim in moved}
    stale: set[int] = set()
    for prim in moved:
        for ancestor in prim.iter_path_reverse():
            if ancestor is not prim:
                stale.add(id(ancestor))
            if ancestor is root:
                break

    for node in PostOrderIter(root):  # children before parents
        if (id(node) not in stale) or (id(node) in moved_ids) or (node.shape is None):
            continue
        child_points = _child_points(node)
        if child_points is not None:
            node.shape = PointCloud(positions=child_points)


def _move_to_positions(
    targets: list[tuple[Primitive, np.ndarray]],
    bead_position: BeadPosition = "centroid",
) -> None:
    """
    Translate each Primitive so its position (per `bead_position`) sits at its
    target position.

    Primitive.rigidly_transform re-aims bonded Connectors, which needs every
    Connector to have a direction. A hierarchy built without coordinates has
    anchor == linker everywhere, so then translate without re-aiming and point
    the bonded Connectors at their neighbors once everything has moved.
    """
    directionless = any(
        not connector.has_bond_vector
        for target, _ in targets
        for connector in target.connectors
    )
    for target, position in targets:
        current = bead_position_of(target, bead_position)
        if current is None:
            # same convention as primitive_to_mbuild: no position means the origin
            target.shape = PointCloud(positions=np.zeros(3, dtype=float))
            current = np.zeros(3, dtype=float)
        offset = position - current
        translation = RigidTransform.from_translation(offset)
        if not directionless:
            target.rigidly_transform(translation)
            continue
        for node in PreOrderIter(target):
            node._rigidly_transform_shape(translation)
        for connector in target.connectors:
            connector.rigidly_transform(translation)

    if not directionless:
        return
    for target, _ in targets:
        for connector in target.connectors:
            if connector.has_neighbor:
                connector.linker.position = np.array(
                    connector.neighbor.anchor.position, dtype=float
                )
                connector.neighbor.linker.position = np.array(
                    connector.anchor.position, dtype=float
                )


def update_primitive_from_mbuild(
    prim: Primitive,
    compound: Compound,
    coords_to_nm: float = 1.0,
    bead_position: BeadPosition = "centroid",
) -> list[Primitive]:
    """Write the positions of an mBuild Compound back onto an existing MuPT hierarchy.

    Each particle (leaf Compound) of `compound` is matched to the Primitive in
    `prim`'s hierarchy that has the same address, which `primitive_to_mbuild`
    stores on every Compound it creates. The matched Primitive is rigidly
    translated (`Primitive.rigidly_transform`) so that its position (centroid or
    center of mass, see `bead_position`) sits at the
    particle's position, which also moves its Connectors and the shapes of any
    Primitives below it. A matched Primitive with no shape is treated as sitting
    at the origin, as `primitive_to_mbuild` exports it. If the hierarchy has no
    coordinates yet (Connectors without a direction), Primitives are moved without
    re-aiming and bonded Connectors are then pointed at their neighbors, which gives
    them a direction. Afterwards, the shapes
    of the Primitives above the moved ones are rebuilt as PointClouds.

    mBuild particles carry positions only, so matched Primitives are translated,
    never rotated: an anisotropic coarse-grained bead keeps its orientation.

    Typical use: export with `primitive_to_mbuild`, relax or simulate the
    Compound's coordinates elsewhere, set them on that same Compound (e.g.
    `compound.xyz = new_positions_nm`), then call this function with the same
    `coords_to_nm` used for the export.

    Parameters
    ----------
    prim : Primitive
        The Primitive that was passed to `primitive_to_mbuild` (or any ancestor of it).
        It is updated in place.
    compound : mbuild.Compound
        A Compound produced by `primitive_to_mbuild`, with updated particle positions.
        Coordinates must already be unwrapped across periodic boundaries.
    coords_to_nm : float, optional, default 1.0
        Multiplication factor that converts the tree's coordinates into nanometers,
        exactly as passed to `primitive_to_mbuild` (e.g., use 0.1 if the tree is in
        Angstrom). Compound positions are divided by it.
    bead_position : {"centroid", "center_of_mass"}, default "centroid"
        How a Primitive's position is measured, exactly as passed to
        `primitive_to_mbuild`: each matched Primitive is moved so that this
        position lands on its particle.

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
    check_bead_position(bead_position)
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
        position = np.asarray(particle.pos, dtype=float) / coords_to_nm
        if not np.all(np.isfinite(position)):
            raise ValueError(
                f"mBuild particle {particle.name!r} has a non-finite "
                f"position {position}"
            )
        targets.append((by_address[address], position))

    moved = [target for target, _ in targets]
    _move_to_positions(targets, bead_position)
    _refresh_ancestor_shapes(prim, moved)

    return moved
