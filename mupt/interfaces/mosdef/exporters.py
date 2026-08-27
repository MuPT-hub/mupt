'''Export a MuPT Primitive hierarchy to an mBuild Compound or GMSO Topology.

This requires that the MuPT user choose which part of the Primitive is leaf-level.

Choosing a resolution means choosing a stopping floor through the Primitive
tree. Descending from the root, every node either stops (becomes one mBuild leaf
particle) or becomes a grouping Compound over its children. Therefore, hierarchy above
the floor is preserved in the emitted mBuild Compound.

The resolutoin methodology itself is format agnostic, this module adapts its
output into an mBuild Compound. The resolution argument is polymorphic and all
forms reduce to a single "stop descending here" trigger:
    None: leaves (fully atomistic / finest stored)
    int: A depth cut
    PrimitiveRole: A role cut (SAAMR levels)
    Callable[[Primitive], bool]: A custom stop predicate
'''

from typing import Callable, Optional, Union

import numpy as np

from mbuild import Compound
from gmso.core.topology import Topology

from ...mupr.primitives import Primitive, PrimitiveRole
from .._shared.topology import (
    ResolutionSpec,
    resolution_predicate,
    resolution_graph,
)


# Type hints for new objects introduced
# ResolutionPredicate answers "should descent stop at this Primitive?".
# It is the same signature anytree expects for the "stop" argument of its
# iterators (PreOrderIter, LevelOrderIter, etc)
# ResolutionSpec is the user-facing form and resolution_predicate() normalizes every
# accepted form into a single ResolutionPredicate.
ResolutionPredicate = Callable[[Primitive], bool]
ResolutionSpec = Union[None, int, PrimitiveRole, ResolutionPredicate]


# mBuild only accepts these bond orders.
_MB_BOND_ORDERS = {0.0, 1.0, 2.0, 3.0, 1.5}


def to_mbuild(
    root: Primitive,
    resolution: ResolutionSpec = None,
    name: Optional[str] = None,
    coords_to_nm: float = 1.0,
) -> Compound:
    '''Export a resolution slice of a MuPT hierarchy as a nested mBuild Compound.


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
    coords_to_nm : float, optional
        Multiplication factor that converts the tree's coordinates into nanometers, which is the
        unit mBuild requires (e.g., use 0.1 to convert from Angstrom to nm).
        Defaults to 1.0

    Returns
    -------
    mbuild.Compound
        Contains a Compound tree that mirrors the Primitive hierarchy down to the
        resolution floor. Nodes above the floor become grouping Compounds, and
        nodes at the floor become mBuild particles (positioned at each node's
        shape.centroid scaled by coords_to_nm, or the origin if a node has no
        shape).
    '''
    # The resolution designation is read as a stopping point, and becomes the
    # bottom of the mBuild hierarchy (particles and bonds).
    should_stop = resolution_predicate(resolution, root)
    particle_of = {} # dict of primid: mBuild Compound; fast lookup later for mBuild bonds.

    def build(prim: Primitive) -> Compound:
        if prim.is_leaf or should_stop(prim):
            if prim.shape is not None:
                pos = np.asarray(prim.shape.centroid, dtype=float) * coords_to_nm
            else:
                pos = np.zeros(3)
            particle = Compound(
                name=prim.element.symbol if prim.is_atom else str(prim.label),
                pos=pos,
                element=prim.element.symbol if prim.is_atom else None,
            )
            particle_of[id(prim)] = particle
            return particle
        # Create the "container" Compound first if not at a leaf-level
        # Repeated for each higher layer of Primitive repr that isn't caught by should_stop
        # This maintains the topology information above the leaf-level in the mBuild Compound
        group = Compound(name=str(prim.label))
        for child in prim.children:
            group.add(build(child))
        return group

    compound = build(root)
    if name is not None:
        compound.name = name

    # bonds come from the _shared/topology.py resolution_graph(), which already resolves each
    # cross-slice connection down to the correct pair of resolution nodes (per ResolutionSpec)
    graph = resolution_graph(root, resolution)
    for node_u, node_v, data in graph.edges(data=True):
        prim_u = graph.nodes[node_u]['primitive']
        prim_v = graph.nodes[node_v]['primitive']
        order = data.get('bond_order')
        compound.add_bond(
            (particle_of[id(prim_u)], particle_of[id(prim_v)]),
            bond_order=order if order in _MB_BOND_ORDERS else None,
        )

    return compound


def to_gmso(
    root: Primitive,
    resolution: ResolutionSpec = None,
    name: Optional[str] = None,
    coords_to_nm: float = 1.0,
) -> Topology:
    '''Export a resolution slice of a MuPT hierarchy as a gmso.core.Topology.

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
    coords_to_nm : float, optional
        Multiplication factor that converts the tree's coordinates into nanometers, which is the
        unit mBuild always assumes (e.g., use 0.1 to convert from Angstrom to nm).
        Defaults to 1.0

    Returns
    -------
    gmso.core.Topology
    '''

    compound = to_mbuild(
        root=root,
        resolution=resolution,
        name=name,
        coords_to_nm=coords_to_nm
    )
    return compound.to_gmso()
