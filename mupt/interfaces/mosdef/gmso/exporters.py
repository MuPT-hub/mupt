"""
Export a MuPT Primitive hierarchy to an mBuild Compound or GMSO Topology.

This requires that the MuPT user choose which part of the
Primitive hierarchy is leaf-level in the mBuild bond graph.
"""

from typing import Optional

from gmso.core.topology import Topology as GMSOTopology

from ..mbuild.exporters import to_mbuild
from ....trees.subselect import NodePredicate
from ....mupr.primitives import Primitive


def to_gmso(
    root: Primitive,
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
    resolution : None | int | PrimitiveRole | Callable[[Primitive], bool]
        Designates the recursion floor whose nodes become the Compound's leaf
        particles. Defaults to None (fully atomistic). An int depth is measured
        relative to root, so resolution=1 always means "root's depth-1 children
        are the beads", whatever subtree root points at.
    name : str, optional
        Name for the returned Compound. Defaults to the root Primitive's label.
    coords_to_nm : float, optional, default 1.0
        Multiplication factor that converts the tree's coordinates
        into nanometers, which is the unit mBuild always assumes

        (e.g., use 0.1 to convert from Angstrom to nm).

    Returns
    -------
    gmso.core.Topology
    """
    compound = to_mbuild(
        root=root, predicate=predicate, name=name, coords_to_nm=coords_to_nm
    )
    return compound.to_gmso()
