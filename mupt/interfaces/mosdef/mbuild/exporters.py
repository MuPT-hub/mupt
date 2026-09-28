"""
Export a MuPT Primitive hierarchy to an mBuild Compound or GMSO Topology.

This requires that the MuPT user choose which part of the
Primitive hierarchy is leaf-level in the mBuild bond graph.
"""

from typing import Optional

import numpy as np
from mbuild.compound import Compound

from ....chemistry.core import ElementLike
from ....mupr.primitives import (
    Primitive,
    PrimitivePredicate,
    indiscriminate_selector,
)
from ....mupr.properties import is_atom


# mBuild only accepts these bond orders.
_MB_BOND_ORDERS: set[float] = {0.0, 1.0, 2.0, 3.0, 1.5}


def to_mbuild(
    root: Primitive,
    predicate: Optional[PrimitivePredicate] = None,
    name: Optional[str] = None,
    coords_to_nm: float = 1.0,
) -> Compound:
    """Export a resolution slice of a MuPT hierarchy as a nested mBuild Compound.


    See https://mbuild.mosdef.org for information on mBuild.

    Parameters
    ----------
    root : Primitive
        Root of the MuPT hierarchy to export.
    resolution : None | int | PrimitiveRole | Callable[[Primitive], bool]
        Designates the recursion floor whose nodes become the Compound's leaf
        particles. Defaults to None (fully atomistic). An int depth is measured
        relative to root, so resolution=1 always means "root's depth-1 children
        are the beads", whatever subtree root points at.
    name : str, optional
        Name for the returned Compound. Defaults to the root Primitive's label.
    coords_to_nm : float, optional, default 1.0
        Multiplication factor that converts the tree's coordinates into nanometers,
        which is the unit mBuild requires (e.g., use 0.1 to convert from Angstrom to nm)

    Returns
    -------
    mbuild.Compound
        Contains a Compound tree that mirrors the Primitive hierarchy down to the
        resolution floor. Nodes above the floor become grouping Compounds, and
        nodes at the floor become mBuild particles (positioned at each node's
        shape.centroid scaled by coords_to_nm, or the origin if a node has no
        shape).
    """
    # The resolution designation is read as a stopping point, and becomes the
    # bottom of the mBuild hierarchy (particles and bonds).
    if predicate is None:
        predicate = indiscriminate_selector

    particle_of: dict[int, Compound] = {}  # fast lookup later for mBuild bonds.

    def build(prim: Primitive) -> Compound:
        if predicate(prim):
            if prim.shape is not None:
                pos = np.asarray(prim.shape.centroid, dtype=float) * coords_to_nm
            else:
                pos = np.zeros(3)

            element: Optional[ElementLike] = None
            element_symbol: Optional[str] = None
            if is_atom(root):
                element = root.element  # TB: linter doesn't pick up on clause above
                element_symbol = element.symbol

            particle = Compound(
                name=(element_symbol or str(prim.label)),
                pos=pos,
                element=element_symbol,
            )
            particle_of[id(prim)] = particle
            return particle

        # Create the "container" Compound first if not at a leaf-level
        # Repeated for each higher layer of mupt repr that isn't caught by should_stop
        # Maintains the topology information above the leaf-level in the mBuild Compound
        group = Compound(name=str(prim.label))
        for child in prim.children:
            group.add(build(child))
        return group

    compound = build(root)
    if name is not None:
        compound.name = name

    graph = root.cross_section(predicate=predicate)
    for node_u, node_v, data in graph.edges(data=True):
        prim_u = graph.nodes[node_u]["primitive"]
        prim_v = graph.nodes[node_v]["primitive"]
        order = data.get("bond_order")
        compound.add_bond(
            (particle_of[id(prim_u)], particle_of[id(prim_v)]),
            bond_order=order if order in _MB_BOND_ORDERS else None,
        )

    return compound
