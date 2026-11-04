"""
Export a MuPT Primitive hierarchy to an mBuild Compound or GMSO Topology.
This requires that the MuPT user choose which part of the
Primitive hierarchy is leaf-level in the mBuild bond graph.
"""

from typing import Optional

import numpy as np
from mbuild.compound import Compound, Particle

from ....mupr.primitives.primitive import Primitive
from ....mupr.primitives.simple.atomic import AtomicPrimitive


# mBuild only accepts these bond orders.
_MB_BOND_ORDERS: set[float] = {0.0, 1.0, 2.0, 3.0, 1.5}


def _primitive_to_mbuild_compound(
    prim: Primitive,
    name: Optional[str] = None,
    coords_to_nm: float = 1.0,
    as_particle: bool = False,
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

    compound_class: type[Compound] = Particle if as_particle else Compound
    compound = compound_class(
        name=name,
        pos=pos,
        element=element_symbol,
    )
    # bookkeeping mechanism for reading coords back in
    setattr(compound, "address", prim.address)

    return compound
