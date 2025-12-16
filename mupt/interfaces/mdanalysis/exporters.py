"""
MuPT to MDAnalysis Topology Exporter

This module provides functionality to convert MuPT Representation objects
(univprim) into MDAnalysis Universe objects, focusing on topology information
(atoms, residues, segments, and bonds).
"""

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

import MDAnalysis as mda
from MDAnalysis.guesser import DefaultGuesser

import numpy as np
from typing import Any, Dict, List, Optional

from ...mupr.primitives import Primitive, PrimitiveHandle

def _pdb_resname(label: str, resname_map: Optional[dict]) -> str:
    if resname_map and label in resname_map:
        name = resname_map[label]
    else:
        name = label

    if len(name) != 3:
        raise ValueError(
            f"Residue name '{name}' (from '{label}') is not 3 characters long"
        )
    return name.upper()

def primitive_to_mdanalysis(univprim : Primitive,
                            resname_map: Optional[dict[str, str]] = None,
                            ) -> mda.Universe:
    """
    Convert a MuPT Representation (univprim) to an MDAnalysis Universe.
    
    This function extracts topology information from a univprim tree structure
    and creates an MDAnalysis Universe with atoms, residues, segments, and bonds.
    Note that coordinates are not included as the univprim may not contain them.
    
    Parameters
    ----------
    univprim : Primitive
        The universal primitive representation containing the molecular system
        in a tree-like hierarchy (universe -> chains -> residues -> atoms).
    resname_map : dict, optional
        A mapping from residue labels to PDB residue names (3-letter codes).
        If provided, this mapping will be used to set residue names in the
        MDAnalysis Universe. If not provided, residue labels from univprim
        will be used directly.

    Returns
    -------
    MDAnalysis.Universe
        A new Universe object containing the topology information extracted
        from the Primitive. The Universe will have:
        - Atoms with elements, names, and unique IDs
        - Residues with names and IDs
        - Segments with IDs
        - Bond connectivity
    
    Raises
    ------
    ValueError
        If the Primitive structure is malformed or missing required attributes.
    AttributeError
        If required methods or attributes are not available on Primitive.
    
    Notes
    -----
    - Atom array indices are 0-based (MDAnalysis internal convention)
    - Atom IDs, residue IDs, and segment IDs are 1-based (user-facing)
    - Bond indices are 0-based (array indices)
    - Atom names are set to element symbols (atom-type agnostic approach)
    - No coordinates are included in the Universe
    
    Examples
    --------
    >>> universe = primitive_to_mdanalysis(my_univprim)
    >>> print(f"Created universe with {universe.atoms.n_atoms} atoms")
    >>> print(f"Number of residues: {universe.residues.n_residues}")
    >>> print(f"Number of segments: {universe.segments.n_segments}")
    """
    
    assert univprim.height >= 3, "Primitive must have at least 3 levels: universe -> chains -> residues -> atoms"

    # ----------------------------
    # Containers (allow duplicates)
    # ----------------------------
    atom_elements = []
    atom_names = []
    atom_positions = []

    atom_resindex = []
    atom_segindex = []

    residue_names = []
    residue_segindex = []
    residue_ids = []

    bonds = []

    # Counters
    atom_idx = 0
    res_idx = 0

    # ----------------------------
    # Traverse hierarchy explicitly
    # ----------------------------
    for chain_idx, chain in enumerate(univprim.children):

        resid_counter = 1  # reset per chain

        for residue in chain.children:
            residue_names.append(
                _pdb_resname(residue.label, resname_map)
            )

            residue_segindex.append(chain_idx)
            residue_ids.append(resid_counter)

            # Local atom index map for this residue only
            local_atom_indices = {}

            for atom in residue.children:
                # Record atom
                atom_elements.append(atom.element.symbol)
                atom_names.append(atom.element.symbol)

                if hasattr(atom, "shape") and atom.shape is not None:
                    atom_positions.append(atom.shape.centroid)
                else:
                    atom_positions.append([0.0, 0.0, 0.0])

                atom_resindex.append(res_idx)
                atom_segindex.append(chain_idx)

                local_atom_indices[atom] = atom_idx
                atom_idx += 1

            # Bonds (local → global index)
            if hasattr(residue, "topology") and residue.topology is not None:
                for a1, a2 in residue.topology.edges():
                    if a1 in local_atom_indices and a2 in local_atom_indices:
                        bonds.append(
                            (local_atom_indices[a1], local_atom_indices[a2])
                        )

            resid_counter += 1
            res_idx += 1

    # ----------------------------
    # Convert to numpy arrays
    # ----------------------------
    atom_positions = np.asarray(atom_positions, dtype=float)
    atom_resindex = np.asarray(atom_resindex, dtype=int)
    atom_segindex = np.asarray(atom_segindex, dtype=int)
    residue_segindex = np.asarray(residue_segindex, dtype=int)

    num_atoms = len(atom_resindex)
    num_residues = len(residue_names)
    num_segments = len(univprim.children)

    print(f"Atoms: {num_atoms}, Residues: {num_residues}, Segments: {num_segments}")
    print(f"Unique atom_resindex: {np.unique(atom_resindex)}")
    print(f"Unique atom_segindex: {np.unique(atom_segindex)}")

    # ----------------------------
    # Create MDAnalysis Universe
    # ----------------------------
    universe = mda.Universe.empty(
        num_atoms,
        n_residues=num_residues,
        n_segments=num_segments,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )

    # ----------------------------
    # Topology attributes
    # ----------------------------
    universe.add_TopologyAttr("name", atom_names)
    universe.add_TopologyAttr("type", atom_elements)
    universe.add_TopologyAttr("element", atom_elements)
    universe.add_TopologyAttr("resname", residue_names)
    universe.add_TopologyAttr("resid", residue_ids)

    segids = [str(i + 1) for i in range(num_segments)]
    universe.add_TopologyAttr("segid", segids)

    if bonds:
        universe.add_TopologyAttr("bonds", np.asarray(bonds, dtype=np.int32))

    universe.atoms.positions = atom_positions

    return universe
