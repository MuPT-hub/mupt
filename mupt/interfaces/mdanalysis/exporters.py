"""
MuPT to MDAnalysis Topology Exporter

This module provides functionality to convert MuPT Primitive hierarchies
into MDAnalysis Universe objects, focusing on topology information
(atoms, residues, segments, and bonds).

The export logic is delegated to pluggable strategy objects
(see :mod:`~mupt.interfaces.mdanalysis.strategies`), allowing
the same entry point to support all-atom, coarse-grained, or
custom export schemes.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

import numpy as np
from typing import Optional
from collections import Counter

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from .strategies import MDAExportStrategy, AllAtomExportStrategy, MDATopologyData
from ...mupr.primitives import Primitive


def _build_mda_universe(data: MDATopologyData) -> mda.Universe:
    """
    Construct an MDAnalysis Universe from pre-collected topology data.

    This is a shared builder that converts the flat arrays in
    :class:`MDATopologyData` into a fully-formed MDAnalysis Universe
    with atoms, residues, segments, bonds, and coordinates.

    Parameters
    ----------
    data : MDATopologyData
        Topology arrays collected by a strategy's ``collect_topology()``.

    Returns
    -------
    mda.Universe
        MDAnalysis Universe with topology attributes and coordinates.
    """
    atom_positions = np.asarray(data.atom_positions, dtype=float)
    atom_resindex = np.asarray(data.atom_resindex, dtype=int)
    residue_segindex = np.asarray(data.residue_segindex, dtype=int)

    num_atoms = len(data.atom_elements)
    num_residues = len(data.residue_names)
    num_segments = data.num_segments

    LOGGER.info(
        f"Atoms: {num_atoms}, Residues: {num_residues}, Segments: {num_segments}"
    )

    # Create empty Universe with correct hierarchy sizes
    universe = mda.Universe.empty(
        num_atoms,
        n_residues=num_residues,
        n_segments=num_segments,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )

    # Topology attributes
    universe.add_TopologyAttr("name", data.atom_names)
    universe.add_TopologyAttr("type", data.atom_elements)
    universe.add_TopologyAttr("element", data.atom_elements)
    universe.add_TopologyAttr("resname", data.residue_names)
    universe.add_TopologyAttr("resid", data.residue_ids)

    segids = [str(i + 1) for i in range(num_segments)]
    universe.add_TopologyAttr("segid", segids)

    # Bonds
    if data.bonds:
        LOGGER.info(f"Adding {len(data.bonds)} bonds to universe")

        if data.bond_orders and len(data.bond_orders) == len(data.bonds):
            bond_attr = Bonds(
                np.asarray(data.bonds, dtype=np.int32), order=data.bond_orders
            )
            universe.add_TopologyAttr(bond_attr)
            LOGGER.info(
                f"Added {len(data.bonds)} bonds with {len(data.bond_orders)} bond orders"
            )

            order_counts = Counter(data.bond_orders)
            LOGGER.info(f"Bond order distribution: {dict(order_counts)}")
        else:
            universe.add_TopologyAttr(
                "bonds", np.asarray(data.bonds, dtype=np.int32)
            )
            LOGGER.info(f"Added {len(data.bonds)} bonds (no bond orders)")
    else:
        LOGGER.info("No bonds to add")

    universe.atoms.positions = atom_positions

    return universe


def primitive_to_mdanalysis(
    univprim: Primitive,
    resname_map: dict[str, str],
    strategy: Optional[MDAExportStrategy] = None,
) -> mda.Universe:
    """
    Convert a MuPT Primitive hierarchy to an MDAnalysis Universe.

    This function delegates topology collection to a pluggable
    :class:`MDAExportStrategy`. When no strategy is provided,
    :class:`AllAtomExportStrategy` is used.

    Parameters
    ----------
    univprim : Primitive
        Root Primitive of the molecular system to export.
    resname_map : dict[str, str]
        Mapping from residue labels to PDB residue names (3-letter codes).
    strategy : MDAExportStrategy, optional
        Export strategy to use. If ``None`` (default),
        :class:`AllAtomExportStrategy` is used.

    Returns
    -------
    mda.Universe
        A new Universe object containing topology information extracted
        from the Primitive.  The Universe will have:

        - Atoms with elements, names, and unique IDs
        - Residues with names and IDs
        - Segments with IDs
        - Bond connectivity

    Raises
    ------
    ValueError
        If the Primitive hierarchy lacks required role assignments.
        Use :func:`~mupt.mupr.roles.assign_SAAMR_roles` for standard
        hierarchies or assign roles manually.

    Notes
    -----
    - Atom array indices are 0-based (MDAnalysis internal convention)
    - Atom IDs, residue IDs, and segment IDs are 1-based (user-facing)
    - Bond indices are 0-based (array indices)
    - Atom names are set to element symbols (atom-type agnostic approach)
    - Coordinates are extracted from atom centroids, defaulting to the
      strategy's ``default_atom_position`` (``[0,0,0]`` unless overridden)
    - The four SAAMR roles (UNIVERSE, SEGMENT, RESIDUE, PARTICLE) are the
      only roles recognized for MDAnalysis export, but the tree may contain
      intermediate nodes at any depth between these roles. Such nodes carry
      ``PrimitiveRole.UNASSIGNED`` and are traversed transparently.

    Examples
    --------
    >>> universe = primitive_to_mdanalysis(my_univprim, resname_map)
    >>> universe.atoms.n_atoms
    42

    Using an explicit strategy:

    >>> from mupt.interfaces.mdanalysis.strategies import AllAtomExportStrategy
    >>> strategy = AllAtomExportStrategy()
    >>> universe = primitive_to_mdanalysis(my_univprim, resname_map, strategy=strategy)
    """
    if strategy is None:
        strategy = AllAtomExportStrategy()

    # Delegate topology collection to the strategy
    data = strategy.collect_topology(univprim, resname_map)

    # Build and return the MDAnalysis Universe
    return _build_mda_universe(data)
