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

import logging

LOGGER = logging.getLogger(__name__)

import numpy as np
from typing import Optional
from collections import Counter

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from .strategies import MDAExportStrategy, AllAtomExportStrategy, MDATopologyData
from ...mupr.primitives.primitive import Primitive


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

    num_atoms = len(data.atom_names)
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
    universe.add_TopologyAttr("type", data.atom_types or data.atom_elements)
    if data.atom_elements:  # absent for coarse-grained beads
        universe.add_TopologyAttr("element", data.atom_elements)
    if data.atom_masses:
        universe.add_TopologyAttr("masses", np.asarray(data.atom_masses, dtype=float))
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
                f"Added {len(data.bonds)} bonds with "
                f"{len(data.bond_orders)} bond orders"
            )

            order_counts = Counter(data.bond_orders)
            LOGGER.info(f"Bond order distribution: {dict(order_counts)}")
        else:
            universe.add_TopologyAttr("bonds", np.asarray(data.bonds, dtype=np.int32))
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
        If no Primitive has role ``"segment"``, a segment has no ``"residue"``
        Primitive, segments or residues are nested in one another, a residue
        has nothing to export, some atoms are outside every segment/residue,
        or a residue name is not 3 characters long.

    Notes
    -----
    - Atom array indices are 0-based (MDAnalysis internal convention)
    - Atom IDs, residue IDs, and segment IDs are 1-based (user-facing)
    - Bond indices are 0-based (array indices)
    - Atom names are set to element symbols (atom-type agnostic approach)
    - Coordinates are extracted from atom centroids, defaulting to the
      strategy's ``default_atom_position`` (``[0,0,0]`` unless overridden)
    - Segments and residues are the Primitives with role ``"segment"`` and
      ``"residue"`` (a ``PrimitiveRole`` member or its string value); they may
      sit at any depth, with untagged grouping Primitives in between.
    - Atoms are written segment by segment and residue by residue, so every
      residue and segment is contiguous, as PDB, GRO and PSF files require.
    - Residue names come from ``residue.metadata["residue_name"]`` if set,
      otherwise from ``resname_map[residue.label]``, otherwise the label.
    - For 1-bead-per-residue output, use
      :class:`~mupt.interfaces.mdanalysis.strategies.CoarseGrainedExportStrategy`.

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
