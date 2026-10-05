"""
Export a MuPT Primitive hierarchy to an mBuild Compound or GMSO Topology.

This requires that the MuPT user choose which part of the
Primitive hierarchy is leaf-level in the mBuild bond graph.
"""

from typing import Optional

import numpy as np
from networkx.classes import Graph
from mbuild.compound import Compound

from ....chemistry.core import BOND_ORDER_ATTR
from ....trees.subselect import NodePredicate
from ....mupr.primitives import Primitive, AtomicPrimitive


# mBuild only accepts these bond orders.
_MB_BOND_ORDERS: set[float] = {0.0, 1.0, 2.0, 3.0, 1.5}


def _primitive_to_mbuild_compound(
    prim: Primitive,
    name: Optional[str] = None,
    coords_to_nm: float = 1.0,
) -> Compound:
    """
    Convert a single Primitive instance and its
    attributes into a single mBuild Compound

    Only transfers attribute data, not connectivity or hierarchy info
    """
    if prim.shape is None:
        pos = np.zeros(3, dtype=float)
    else:
        pos = np.asarray(prim.shape.centroid, dtype=float) * coords_to_nm

    element_symbol: Optional[str] = None
    # if is_atom(prim):
    if isinstance(prim, AtomicPrimitive):
        # TB: would prefer to use 'is_atom(prim)", but linter
        # isn't smart enough to recognize it as equivalent here
        element_symbol = prim.element.symbol

    if name is None:
        name = element_symbol or str(prim.label)

    compound = Compound(
        name=name,
        pos=pos,
        element=element_symbol,
    )
    # bookkeeping mechanism for reading coords back in
    setattr(compound, "address", prim.address)

    return compound


def primitive_to_mbuild(
    prim: Primitive,
    predicate: Optional[NodePredicate[Primitive]] = None,
    name: Optional[str] = None,
    coords_to_nm: float = 1.0,
) -> Compound:
    """Export a resolution slice of a MuPT hierarchy as a nested mBuild Compound.

    See https://mbuild.mosdef.org for information on mBuild.

    Parameters
    ----------
    root : Primitive
        Root of the MuPT hierarchy to export.
    predicate: Optional[NodePredicate[Primitive]] = None,
        The criterion to use to query for sub-primitives which are
        represented in the 'slice' of the representation exported to mbuild
    name : str, optional
        Name for the returned Compound. Defaults to the root Primitive's label.
    coords_to_nm : float, optional, default 1.0
        Multiplication factor that converts the tree's coordinates into nanometers,
        which is the unit mBuild requires (e.g., use 0.1 to convert from Angstrom to nm)

    Returns
    -------
    mbuild.Compound
        A Compound tree that mirrors the Primitive hierarchy down to the resolution.
        Nodes singled out by the passed predicate become mBuild particles.

        Compounds are positioned at each node's shape.centroid scaled by
        coords_to_nm, or the origin ([0., 0., 0.]) if a node has no shape).
    """
    if predicate is None:
        # TB TODO: add support for complete hierarchy export if no predicate is provided
        raise NotImplementedError(
            "Full-hierarchy export to mbuil not yet supported;"
            "please suply a predicate tp sub-select a cross-section to export"
        )

    particle_of: dict[int, Compound] = {}  # fast lookup later for mBuild bonds.
    compound_group = _primitive_to_mbuild_compound(
        prim,
        name=name,
        coords_to_nm=coords_to_nm,
    )

    cross_section: Graph = prim.cross_section(predicate)
    for subprim in cross_section:
        subcompound = _primitive_to_mbuild_compound(subprim, coords_to_nm=coords_to_nm)
        particle_of[subprim.address] = subcompound
        compound_group.add(subcompound)

    for subprim0, subprim1, data in cross_section.edges(data=True):
        bond_order = data.get(BOND_ORDER_ATTR)
        if bond_order not in _MB_BOND_ORDERS:
            bond_order = None

        compound_group.add_bond(
            (particle_of[subprim0.address], particle_of[subprim1.address]),
            bond_order=bond_order,
        )

    return compound_group
