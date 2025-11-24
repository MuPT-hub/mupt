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



def primitive_to_mdanalysis(univprim : Primitive,
                            coords: Optional[np.ndarray] = None) -> mda.Universe:
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
    coords : np.ndarray, optional
        An optional array of shape (N, 3) containing atomic coordinates. If provided,
        these coordinates will be assigned to the Universe. If None, no coordinates
        will be set.
    
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

    num_residues = 0
    num_chains = 0
    num_atoms = 0

    chain_idx_counter = 0
    residue_idx_counter = 0
    atom_idx_counter = 0

    chain_idx_arr = []
    residue_idx_arr = []
    atom_idx_arr = []

    residue_chain_index = []

    atom_residue_resname = [] # one value per atom, corresponds to the residue name
    atom_element_identifier = [] # one value per atom, corresponds to the element type
    atom_name_identifier = [] # one value per atom, corresponds to the atom name

    residue_resname_mapper = [] # one value per residue, corresponds to the residue name

    # for every chain x in the universe
    for x in univprim.children_by_handle.keys():
        # for every residue y in chain x
        num_chains += 1

        for y in univprim.children_by_handle[x].children_by_handle.keys():
            num_residues += 1
            residue_chain_index.append(chain_idx_counter)

            residue = univprim.children_by_handle[x].children_by_handle[y]

            residue_resname_mapper.append(residue.label)

            # for every atom z in residue y of chain x
            for z in residue.children_by_handle.keys():
                atom = residue.children_by_handle[z]
                num_atoms += 1

                chain_idx_arr.append(chain_idx_counter) # 0-indexed
                residue_idx_arr.append(residue_idx_counter) # 0-indexed
                atom_idx_arr.append(atom_idx_counter) # 0-indexed
                atom_residue_resname.append(residue.label)

                # Extract element
                if not hasattr(atom, 'element') or atom.element is None:
                    raise ValueError(f"Atom at index {atom_idx_counter} is missing element information")
                
                atom_element_identifier.append(atom.element.symbol)
                atom_name_identifier.append(atom.element.symbol)  # Use element symbol as atom name

                atom_idx_counter += 1
            
            residue_idx_counter += 1

        chain_idx_counter += 1

    chain_idx_arr = np.array(chain_idx_arr, dtype=int)
    residue_idx_arr = np.array(residue_idx_arr, dtype=int)
    atom_idx_arr = np.array(atom_idx_arr, dtype=int)
    residue_chain_index = np.array(residue_chain_index, dtype=int)


    assert len  (chain_idx_arr) == num_atoms
    assert len  (residue_idx_arr) == num_atoms
    assert len  (atom_idx_arr) == num_atoms

    assert len  (atom_residue_resname) == num_atoms
    assert len  (atom_element_identifier) == num_atoms
    assert len  (atom_name_identifier) == num_atoms

    assert len (residue_chain_index) == num_residues

    print(f"Total chains: {num_chains}, residues: {num_residues}, atoms: {num_atoms}")

    # Create empty Universe
    universe = mda.Universe.empty(
        num_atoms,
        n_residues=num_residues,
        n_segments=num_chains,
        atom_resindex=residue_idx_arr,
        residue_segindex=residue_chain_index,
        trajectory=True  # For storing coordinates
    )

    # Add topology attributes
    
    # Atom names (using element symbols)
    universe.add_TopologyAttr('name', atom_name_identifier)

    # Atom types (using element symbols)
    universe.add_TopologyAttr('type', atom_element_identifier)

    # Elements
    universe.add_TopologyAttr('element', atom_element_identifier)

    # Residue names
    universe.add_TopologyAttr('resname', residue_resname_mapper)

    # Residue IDs (1-based)
    resids = list(range(1, num_residues + 1))
    universe.add_TopologyAttr('resid', resids)

    # Segment IDs (1-based, as strings)
    segids = [str(i) for i in range(1, num_chains + 1)]
    universe.add_TopologyAttr('segid', segids)

    guesser = DefaultGuesser(universe, fudge_factor=1.2)

    if coords is not None:
        if coords.shape != (num_atoms, 3):
            raise ValueError(f"Provided coordinates shape {coords.shape} does not match number of atoms {num_atoms}")
        universe.atoms.positions = coords

        universe.guess_TopologyAttrs(to_guess=["types", 
                                               "bonds", "angles", "dihedrals", 
                                               "impropers", "aromaticities",
                                               "masses"],
                        context=guesser, fudge_factor=0.5)
        
    else:
        universe.guess_TopologyAttrs(
                            to_guess=["types",  "masses"], # can only safely guess types and masses
                      context=guesser, fudge_factor=0.5)
        
    return universe

def debug_topology_mapping(univprim) -> None:
    """
    Print detailed information about how atoms map to residues and segments.
    
    This is a debugging utility to verify that the topology hierarchy is
    constructed correctly.
    
    Parameters
    ----------
    univprim : Primitive
        The primitive representation to debug.
    
    Examples
    --------
    >>> debug_topology_mapping(univprim)
    Chain 0: 3 residues
    Residue 0 (ALA): 5 atoms
    Residue 1 (GLY): 4 atoms
    ...
    """
    print("=== Topology Hierarchy Debug ===")
    print(f"Total chains: {len(univprim.children)}")
    print(f"Hierarchy depth: {univprim.height}")
    
    if univprim.height == 4:
        print("Structure: Universe -> Chains -> Residues -> Substructures -> Atoms\n")
    elif univprim.height == 3:
        print("Structure: Universe -> Chains -> Residues -> Atoms\n")
    else:
        print(f"Warning: Unexpected hierarchy depth of {univprim.height}\n")
    
    atom_counter = 0
    residue_counter = 0
    
    for chain_idx, chain in enumerate(univprim.children):
        chain_label = chain.label if hasattr(chain, 'label') else f"chain_{chain_idx}"
        print(f"Chain {chain_idx} ('{chain_label}'): {len(chain.children)} residues")
        
        for res_idx, residue in enumerate(chain.children):
            res_label = residue.label if hasattr(residue, 'label') else f"res_{res_idx}"
            
            if univprim.height == 4:
                # Count atoms across all substructures
                n_atoms = sum(len(sub.children) for sub in residue.children)
                n_substructures = len(residue.children)
                if n_atoms > 0:
                    print(f"  Residue {residue_counter} ('{res_label}'): {n_atoms} atoms across {n_substructures} substructures (global atom indices {atom_counter} to {atom_counter + n_atoms - 1})")
                    atom_counter += n_atoms
                    residue_counter += 1
            else:
                n_atoms = len(residue.children)
                if n_atoms > 0:
                    print(f"  Residue {residue_counter} ('{res_label}'): {n_atoms} atoms (global atom indices {atom_counter} to {atom_counter + n_atoms - 1})")
                    atom_counter += n_atoms
                    residue_counter += 1
        
        print()  # Blank line between chains
    
    print(f"Total atoms: {atom_counter}")
    print(f"Total residues: {residue_counter}")
    print("="*40)