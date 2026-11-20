"""
Export a MuPT Primitive hierarchy to an mBuild Compound or GMSO Topology.
This requires that the MuPT user choose which part of the
Primitive hierarchy is leaf-level in the mBuild bond graph.
"""

from typing import Optional

from gmso.core.topology import Topology as GMSOTopology

from ..mbuild.exporters import primitive_to_mbuild
from ....trees.subselect import NodePredicate
from ....mupr.primitives import Primitive


def to_gmso(
    prim: Primitive,
    predicate: Optional[NodePredicate[Primitive]] = None,
    name: Optional[str] = None,
    coords_to_nm: float = 1.0,
) -> GMSOTopology:
    """Export a resolution slice of a MuPT hierarchy as a gmso.core.Topology.
    See https://gmso.mosdef.org for information on how to use GMSO to perform
    atom typing, apply a force field, and interfacing with multiple simulation
    engines including LAMMPS, GROMACS, and HOOMD-Blue.

    Parameters
    ----------
    root : Primitive
        Root of the MuPT hierarchy to export.
    predicate : NodePredicate[Primitive]
        The criterion to use to query for sub-primitives which are represented in
        the 'slice' (cross section) of the representation exported to GMSO
    name : str, optional
        Name for the intermediate mBuild Compound. Defaults to the label of `prim`.
    coords_to_nm : float, optional, default 1.0
        Multiplication factor that converts the tree's coordinates
        into nanometers, which is the unit mBuild always assumes
        (e.g., use 0.1 to convert from Angstrom to nm).

    Returns
    -------
    gmso.core.Topology
    """
    compound = primitive_to_mbuild(
        prim=prim, predicate=predicate, name=name, coords_to_nm=coords_to_nm
    )
    return compound.to_gmso()
