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
from MDAnalysis.core.topologyattrs import Bonds

import numpy as np
from typing import Any, Dict, List, Optional

from ...mupr.primitives import Primitive, PrimitiveHandle
from ...chemistry.core import BOND_ORDER

# DEV:JRL -> Debating whether to keep resname_map optional.
def _pdb_resname(label: str, resname_map: Optional[dict]) -> str:
    """
    Map a residue label to a PDB-compliant 3-character residue name.

    This helper function is used to ensure residue names are valid for 
    PDB export and downstream visualization (e.g., in PyMOL). 
    It optionally applies a user-provided mapping from original labels 
    to 3-letter codes and enforces uppercase formatting.

    Parameters
    ----------
    label : str
        Original residue label from the Primitive object.
    resname_map : dict, optional
        Optional mapping from residue labels to 3-character PDB residue names. 
        If the label is in the dictionary, the mapped value is used. 
        Otherwise, the original label is returned.

    Returns
    -------
    str
        Uppercase, 3-character PDB-compliant residue name.

    Raises
    ------
    ValueError
        If the resulting residue name is not exactly 3 characters long.

    Examples
    --------
    >>> _pdb_resname('head', {'head': 'HEA', 'tail': 'TAL'})
    'HEA'
    >>> _pdb_resname('mid', {'head': 'HEA', 'tail': 'TAL'})
    'MID'
    """
    if resname_map and label in resname_map:
        name = resname_map[label]
    else:
        name = label

    if len(name) != 3:
        raise ValueError(
            f"Residue name '{name}' (from '{label}') is not 3 characters long"
        )
    return name.upper()

def _is_AA_export_compliant(prim : Primitive) -> bool:
    """
    Check whether a Primitive hierarchy is organized
    as universe -> chain -> residue -> atom
    """   
    
    return all(
        leaf.is_atom and (leaf.depth == 3)        
            for leaf in prim.leaves
    )

# DEV:JRL -> Debating whether to keep resname_map optional.
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
        will be used directly. NOTE: MDAnalysis allows for arbitrary length
        residue names, and will automatically truncate the names on export.

    Returns
    -------
    MDAnalysis.Universe
        A new Universe object containing the topology information extracted
        from the Primitive. The Universe will have:
        - Atoms with elements, names, and unique IDs
        - Residues with names and IDs
        - Segments with IDs
        - Bond connectivity
    
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
    
    assert _is_AA_export_compliant(univprim), "[Under Construction] Primitive must be ordered according to: universe -> chains -> residues -> atoms"

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
    bond_orders = []  # Track bond orders for each bond
    bonds_set = set()  # Track to avoid duplicates

    # Maps to translate between atom hierarchy and global indices
    # Structure: residue_atom_maps[global_res_idx] = list of atom handles in that residue (in order)
    residue_atom_maps = []  # List of lists: each inner list is atom handles for that residue

    # We also need a way to map (chain_idx, residue_handle, atom_handle) -> global atom index
    # But since atom handles like ('ATOM', 14) are unique WITHIN a residue, we need more info
    # Strategy: Store both the residue object and its atom handle-to-global-index mapping
    residue_to_atom_global_idx = []  # List of dicts: residue_to_atom_global_idx[global_res_idx][atom_handle] = global_atom_idx

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

            # Build mapping from atom handle -> global index for this residue
            atom_handle_to_global = {}
            local_atoms_in_residue = []

            for atom_handle, atom in residue.children_by_handle.items():
                # Record atom
                atom_elements.append(atom.element.symbol)
                atom_names.append(atom.element.symbol)

                if hasattr(atom, "shape") and atom.shape is not None:
                    atom_positions.append(atom.shape.centroid)
                else:
                    atom_positions.append([0.0, 0.0, 0.0])

                atom_resindex.append(res_idx)
                atom_segindex.append(chain_idx)

                # Map atom handle to global index
                atom_handle_to_global[atom_handle] = atom_idx
                local_atoms_in_residue.append(atom_handle)

                atom_idx += 1

            # Store mappings for this residue
            residue_atom_maps.append(local_atoms_in_residue)
            residue_to_atom_global_idx.append(atom_handle_to_global)

            # Bonds (local → global index) - INTRA-RESIDUE
            # For intra-residue bonds, we need to find the bond order from internal_connections
            if hasattr(residue, "topology") and residue.topology is not None:
                for atom_handle_1, atom_handle_2 in residue.topology.edges():
                    if atom_handle_1 in atom_handle_to_global and atom_handle_2 in atom_handle_to_global:
                        global_idx_1 = atom_handle_to_global[atom_handle_1]
                        global_idx_2 = atom_handle_to_global[atom_handle_2]
                        
                        bond_pair = tuple(sorted([global_idx_1, global_idx_2]))
                        if bond_pair not in bonds_set:
                            bonds.append(bond_pair)
                            bonds_set.add(bond_pair)
                            
                            # Get bond order from residue's internal_connections
                            bond_order = 1.0  # Default to single bond
                            if hasattr(residue, "internal_connections"):
                                for conn_ref1, conn_ref2 in residue.internal_connections:
                                    if (conn_ref1.primitive_handle == atom_handle_1 and conn_ref2.primitive_handle == atom_handle_2) or \
                                    (conn_ref1.primitive_handle == atom_handle_2 and conn_ref2.primitive_handle == atom_handle_1):
                                        # Found the connection, get the connector and its bondtype
                                        atom1 = residue.fetch_child(atom_handle_1)
                                        # Use .connectors which is a UniqueRegistry that can be indexed by handle
                                        if conn_ref1.connector_handle in atom1.connectors:
                                            connector = atom1.connectors[conn_ref1.connector_handle]
                                            if hasattr(connector, 'bondtype') and connector.bondtype in BOND_ORDER:
                                                bond_order = BOND_ORDER[connector.bondtype]
                                        break
                            bond_orders.append(bond_order)

            resid_counter += 1
            res_idx += 1

    # ----------------------------
    # INTER-RESIDUE BONDS
    # ----------------------------
    # Now process inter-residue bonds stored at the chain level
    # We need to map residue handles to their global residue index

    print(f"Processing inter-residue bonds...")
    inter_residue_bond_count = 0

    for chain_idx, chain in enumerate(univprim.children):
        # Build mapping from residue handle -> global residue index for this chain
        residue_handle_to_global_res_idx = {}
        global_res_offset = sum(len(univprim.children[c].children) for c in range(chain_idx))
        
        for local_res_idx, (res_handle, residue) in enumerate(chain.children_by_handle.items()):
            residue_handle_to_global_res_idx[res_handle] = global_res_offset + local_res_idx
        
        if hasattr(chain, "internal_connections") and chain.internal_connections:
            for conn_ref1, conn_ref2 in chain.internal_connections:
                residue1_handle = conn_ref1.primitive_handle
                residue2_handle = conn_ref2.primitive_handle
                
                try:
                    residue1 = chain.fetch_child(residue1_handle)
                    residue2 = chain.fetch_child(residue2_handle)
                except Exception as e:
                    print(f"  Could not fetch residues: {e}")
                    continue
                
                # The connector handles in conn_ref1/conn_ref2 are on the RESIDUE level
                # We need to look up residue.external_connectors to find the ATOM handle
                
                if conn_ref1.connector_handle not in residue1.external_connectors:
                    print(f"  Connector {conn_ref1.connector_handle} not in residue1.external_connectors")
                    continue
                if conn_ref2.connector_handle not in residue2.external_connectors:
                    print(f"  Connector {conn_ref2.connector_handle} not in residue2.external_connectors")
                    continue
                
                # Get the atom references
                atom_ref1 = residue1.external_connectors[conn_ref1.connector_handle]
                atom_ref2 = residue2.external_connectors[conn_ref2.connector_handle]
                
                # atom_ref1.primitive_handle is the ATOM handle within residue1
                atom1_handle = atom_ref1.primitive_handle
                atom2_handle = atom_ref2.primitive_handle
                
                # Get global residue indices
                global_res1_idx = residue_handle_to_global_res_idx[residue1_handle]
                global_res2_idx = residue_handle_to_global_res_idx[residue2_handle]
                
                # Get global atom indices
                if atom1_handle in residue_to_atom_global_idx[global_res1_idx]:
                    global_atom1_idx = residue_to_atom_global_idx[global_res1_idx][atom1_handle]
                else:
                    print(f"  Atom handle {atom1_handle} not found in residue {global_res1_idx}")
                    continue
                    
                if atom2_handle in residue_to_atom_global_idx[global_res2_idx]:
                    global_atom2_idx = residue_to_atom_global_idx[global_res2_idx][atom2_handle]
                else:
                    print(f"  Atom handle {atom2_handle} not found in residue {global_res2_idx}")
                    continue
                
                bond_pair = tuple(sorted([global_atom1_idx, global_atom2_idx]))
                if bond_pair not in bonds_set:
                    bonds.append(bond_pair)
                    bonds_set.add(bond_pair)
                    inter_residue_bond_count += 1
                    
                    # Get bond order from the connector on residue1
                    # Use .connectors which is a UniqueRegistry that can be indexed by handle
                    bond_order = 1.0  # Default to single bond
                    if conn_ref1.connector_handle in residue1.connectors:
                        connector = residue1.connectors[conn_ref1.connector_handle]
                        if hasattr(connector, 'bondtype') and connector.bondtype in BOND_ORDER:
                            bond_order = BOND_ORDER[connector.bondtype]
                    bond_orders.append(bond_order)

    print(f"Added {inter_residue_bond_count} inter-residue bonds")
    print(f"Total bonds: {len(bonds)}")
    print(f"Bond orders collected: {len(bond_orders)}")

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
    print(f"Total bonds collected: {len(bonds)}")
    print(f"Total bond orders collected: {len(bond_orders)}")

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
        print(f"Adding {len(bonds)} bonds to universe")
        
        # Add bond orders if available - must be passed during bonds creation
        if bond_orders and len(bond_orders) == len(bonds):
            # MDAnalysis Bonds class accepts order= parameter at creation time
            bond_attr = Bonds(np.asarray(bonds, dtype=np.int32), order=bond_orders)
            universe.add_TopologyAttr(bond_attr)
            print(f"Added {len(bonds)} bonds with {len(bond_orders)} bond orders")
            
            # Show distribution of bond orders
            from collections import Counter
            order_counts = Counter(bond_orders)
            print(f"Bond order distribution: {dict(order_counts)}")
        else:
            universe.add_TopologyAttr("bonds", np.asarray(bonds, dtype=np.int32))
            print(f"Added {len(bonds)} bonds (no bond orders)")
    else:
        print("No bonds to add")

    universe.atoms.positions = atom_positions

    return universe
