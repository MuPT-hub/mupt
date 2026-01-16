'''Writers which convert the MuPT molecular representation out to RDKit Mols'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'


from typing import Optional, Dict, Tuple

import numpy as np

from rdkit.Chem.rdchem import (
    Atom,
    Bond,
    Mol,
    RWMol,
    Conformer,
    AtomPDBResidueInfo,
)
from rdkit.Geometry import Point3D
from rdkit import Chem

from .rdprops import RDPropType, assign_property_to_rdobj
from .labelling import RDMOL_NAME_WRITE_PROP
from ... import TOOLKIT_NAME
from ...geometry.arraytypes import Shape
from ...chemistry.conversion import element_to_rdkit_atom
from ...mupr.connection import Connector
from ...mupr.primitives import Primitive, PrimitiveHandle
from ...mutils.allatomutils import _is_AA_export_compliant

def rdkit_atom_from_atomic_primitive(atomic_primitive : Primitive) -> Atom:
    '''Convert an atomic Primitive to an RDKit Atom'''
    if not (atomic_primitive.is_atom and atomic_primitive.is_simple):
        raise ValueError('Cannot export non-atomic Primitive to RDKit Atom')
    
    atom = element_to_rdkit_atom(atomic_primitive.element)
    # TODO: decide how (if at all) to handle aromaticity and stereo
    for key, value in atomic_primitive.metadata.items():
        assign_property_to_rdobj(atom, key, value, preserve_type=True)
    
    return atom

def primitive_to_rdkit(
    primitive : Primitive,
    default_atom_position : Optional[np.ndarray[Shape[3], float]]=None,
) -> Mol:
    '''
    Convert a Primitive hierarchy to an RDKit Mol
    Will return as single Mol instance, even is underlying Primitive represents a collection of multiple disconnected molecules
    
    Will set spatial positions for each atom ("default_atom_position" if not assigned per atom) to a Conformer bound to the returned Mol
    '''
    if default_atom_position is None:
        # DEV: opted to not make this a call to geometry.reference.origin() to decrease coupling and allow choice for differently-determined default down the line
        default_atom_position = np.array([0.0, 0.0, 0.0], dtype=float) 
    if default_atom_position.shape != (3,):
        raise ValueError('Default atom position must be a 3-dimensional vector')
    
    if not primitive.is_atomizable: # TODO: include provision (when no flattening is performed) to preserve atom order with Primitive handle indices
        raise ValueError('Cannot export Primitive with non-atomic parts to RDKit Mol')
    primitive = primitive.flattened() # collapse hierarchy out-of-place to avoid mutating original
    
    ## DEV: modelled assembly in part by OpenFF RDKit TK wrapper
    ## https://github.com/openforcefield/openff-toolkit/blob/5b4941c791cd49afbbdce040cefeb23da298ada2/openff/toolkit/utils/rdkit_wrapper.py#L2330
    
    # 0) prepare Primitive source and RDKit destination
    mol = RWMol()
    conf = Conformer(primitive.num_children + primitive.functionality) # preallocate space for all atoms (including linkers)
    atom_idx_map : dict[int, PrimitiveHandle] = {}
    
    ## special case for atomic Primitives; easier to contract into hierarchy containing that single atom as child (less casework)
    temp_prim : Optional[Primitive] = None
    lone_atom_label : Optional[PrimitiveHandle] = None
    if primitive.is_atom: 
        if primitive.parent is not None:
            raise NotImplementedError('Export for unisolated atomic Primitives (i.e. with pre-existing parents) is not supported')
        temp_prim = Primitive(label='temp')
        lone_atom_label : PrimitiveHandle = temp_prim.attach_child(primitive)
            
    # 1) insert atoms
    for handle, child_prim in primitive.children_by_handle.items(): # at this point after flattening, all children should be atomic Primitives
        assert child_prim.is_atom
        child_prim.check_valence()

        idx : int = mol.AddAtom(rdkit_atom_from_atomic_primitive(child_prim))
        atom_idx_map[handle] = idx
        conf.SetAtomPosition(
            idx,
            child_prim.shape.centroid if (child_prim.shape is not None) else default_atom_position[:],
        ) # geometric centroid is defined for all BoundedShape subtypes
    
    # 2) insert bonds
    ## 2a) bonds from internal connections
    for (conn_ref1, conn_ref2) in primitive.internal_connections:
        atom_idx1 : int = atom_idx_map[conn_ref1.primitive_handle]
        conn1 : Connector = primitive.fetch_connector_on_child(conn_ref1)
        
        atom_idx2 : int = atom_idx_map[conn_ref2.primitive_handle]    
        conn2 : Connector = primitive.fetch_connector_on_child(conn_ref2)
        
        # DEV: bondtypes must be compatible, so will take first for now (TODO: find less order-dependent way of accessing bondtype)
        new_num_bonds : int = mol.AddBond(atom_idx1, atom_idx2, order=conn1.bondtype) 
        bond_metadata : dict[str, RDPropType] = {
            **conn1.metadata,
            **conn2.metadata,
        }
        for bond_key, bond_value in bond_metadata.items():
            assign_property_to_rdobj(mol.GetBondBetweenAtoms(atom_idx1, atom_idx2), bond_key, bond_value, preserve_type=True)
    
    ## 2b) insert and bond linker atoms for each external Connector
    for conn_ref in primitive.external_connectors.values(): # TODO: generalize to work for atomic (i.e. non-hierarchical) Primitives w/o external_connectors
        linker_atom = Atom(0) 
        linker_idx : int = mol.AddAtom(linker_atom) # TODO: transpose metadata from external Connector onto 0-number RDKit Atom
        conn : Connector = primitive.fetch_connector_on_child(conn_ref)
        
        mol.AddBond(atom_idx_map[conn_ref.primitive_handle], linker_idx, order=conn.bondtype)
        conf.SetAtomPosition(linker_idx, conn.linker.position) # TODO: decide whether unset position (e.g. as NANs) should be supported
        # if conn.has_linker_position: # NOTE: this "if" check not done in-line, as conn.linker_position raises AttributeError is unset
            # conf.SetAtomPosition(linker_idx, conn.linker_position)
        # else:
            # conf.SetAtomPosition(linker_idx, default_atom_position[:])
            
    ## 3) transfer Primitive-level metadata (atom metadata should already be transferred)
    assign_property_to_rdobj(mol, 'origin', TOOLKIT_NAME, preserve_type=True) # mark MuPT export for provenance
    for key, value in primitive.metadata.items():
        assign_property_to_rdobj(mol, key, value, preserve_type=True) 
    
    # 4) cleanup
    if not ((temp_prim is None) or (lone_atom_label is None)):
        primitive.detach_child(lone_atom_label)
    conformer_idx : int = mol.AddConformer(conf, assignId=True) # DEV: return this index?
    
    mol = Mol(mol) # freeze writable Mol before returning
    if primitive.label is not None:
        mol.SetProp(RDMOL_NAME_WRITE_PROP, primitive.label)
    
    return mol
    

def primitive_to_rdkit_hierarchical(
    univprim: Primitive,
    resname_map: Dict[str, str],
    default_atom_position: Optional[np.ndarray[Shape[3], float]]=None,
) -> Mol:
    """
    Convert a Primitive hierarchy to an RDKit Mol WITHOUT flattening.
    
    This function traverses the hierarchy explicitly (Universe -> Chains -> Residues -> Atoms)
    and builds an RDKit molecule while preserving:
    - Bond connectivity (both intra-residue and inter-residue)
    - Bond orders from connector bondtypes
    - 3D coordinates
    - Chain/residue metadata as molecule properties
    
    Parameters
    ----------
    univprim : Primitive
        The root primitive (universe level) containing chains -> residues -> atoms
    resname_map : dict[str, str]
        Mapping from residue labels to 3-character names
    default_atom_position : np.ndarray, optional
        Default position for atoms without geometry (default: [0, 0, 0])
    
    Returns
    -------
    rdkit.Chem.Mol
        RDKit molecule with correct bond orders and 3D coordinates
    """

    assert _is_AA_export_compliant(univprim), "[Under Construction] Primitive must be ordered according to: universe -> chains -> residues -> atoms"

    if default_atom_position is None:
        default_atom_position = np.array([0.0, 0.0, 0.0], dtype=float)
    
    # Prepare RDKit molecule
    mol = RWMol()
    
    # Track global atom indices and mappings
    # Key: (chain_idx, res_idx, atom_handle) -> global RDKit atom index
    atom_handle_to_rdkit_idx: Dict[Tuple[int, int, PrimitiveHandle], int] = {}
    
    # For inter-residue bond lookup: (chain_idx, res_handle) -> (global_res_idx, residue_prim)
    residue_info: Dict[Tuple[int, PrimitiveHandle], Tuple[int, Primitive]] = {}
    
    # Per-residue atom handle -> rdkit idx mapping for inter-residue bonds
    # residue_atom_maps[global_res_idx][atom_handle] = rdkit_atom_idx
    residue_atom_maps: list[Dict[PrimitiveHandle, int]] = []
    
    # Metadata for SDF properties
    chain_labels = []
    residues_per_chain = []
    
    # Count total atoms for conformer pre-allocation
    total_atoms = sum(
        len(residue.children)
        for chain in univprim.children
        for residue in chain.children
    )
    conf = Conformer(total_atoms)
    
    global_res_idx = 0
    rdkit_atom_idx = 0
    
    # ========================================
    # PASS 1: Add all atoms to RDKit molecule
    # ========================================
    for chain_idx, chain in enumerate(univprim.children):
        chain_labels.append(chain.label or f"chain_{chain_idx}")
        residues_per_chain.append(len(chain.children))
        
        for res_local_idx, (res_handle, residue) in enumerate(chain.children_by_handle.items()):
            # Store residue info for inter-residue bond processing
            residue_info[(chain_idx, res_handle)] = (global_res_idx, residue)
            
            # Map for this residue's atoms
            atom_map_for_residue: Dict[PrimitiveHandle, int] = {}
            
            # Compute PDB-style residue name (3-char, uppercase)
            if resname_map and residue.label in resname_map:
                pdb_resname = resname_map[residue.label].upper()
            else:
                # Fallback: use first 3 chars of label, uppercase
                pdb_resname = (residue.label or "UNK")[:3].upper()
            
            # Chain ID: use chain label's first char, or letter from index
            chain_label = chain.label or f"chain_{chain_idx}"
            if len(chain_label) == 1 and chain_label.isalpha():
                chain_id = chain_label.upper()
            else:
                # Generate chain ID from index (A, B, C, ... Z, AA, AB, ...)
                chain_id = chr(ord('A') + (chain_idx % 26))
                if chain_idx >= 26:
                    chain_id = chr(ord('A') + (chain_idx // 26) - 1) + chain_id
            
            # Residue ID (1-based, per chain)
            resid = res_local_idx + 1
            
            for atom_handle, atom_prim in residue.children_by_handle.items():
                # Create RDKit atom
                rdkit_atom = element_to_rdkit_atom(atom_prim.element)
                
                # Transfer atom metadata from Primitive
                for key, value in atom_prim.metadata.items():
                    if isinstance(value, (int, float, str)):
                        if isinstance(value, int):
                            rdkit_atom.SetIntProp(key, value)
                        elif isinstance(value, float):
                            rdkit_atom.SetDoubleProp(key, value)
                        else:
                            rdkit_atom.SetProp(key, str(value))
                
                # ==========================================
                # Create PDB residue info for file export
                # This enables proper PDB output with residue/chain info
                # ==========================================
                # Generate atom name: element symbol + index within residue (e.g., "C1", "N2")
                atom_count_in_res = len(atom_map_for_residue)
                atom_name = f"{atom_prim.element.symbol}{atom_count_in_res + 1}"
                # Pad to 4 characters for PDB format (left-padded for 1-char elements)
                if len(atom_prim.element.symbol) == 1:
                    atom_name_pdb = f" {atom_name:<3}"  # e.g., " C1 " for carbon
                else:
                    atom_name_pdb = f"{atom_name:<4}"   # e.g., "CL1 " for chlorine
                
                pdb_info = AtomPDBResidueInfo(
                    atomName=atom_name_pdb,
                    serialNumber=rdkit_atom_idx + 1,  # 1-based serial number
                    residueName=pdb_resname,
                    residueNumber=resid,
                    chainId=chain_id,
                    isHeteroAtom=True,  # Mark as HETATM for non-standard residues
                )
                rdkit_atom.SetMonomerInfo(pdb_info)
                
                # Also store as atom properties for in-memory access (backup)
                rdkit_atom.SetProp("_TriposAtomName", atom_name)
                rdkit_atom.SetProp("_TriposResName", pdb_resname)
                rdkit_atom.SetIntProp("_TriposResNum", resid)
                rdkit_atom.SetProp("_TriposChainId", chain_id)
                rdkit_atom.SetProp("residue_name", pdb_resname)
                rdkit_atom.SetIntProp("residue_id", resid)
                rdkit_atom.SetProp("chain_id", chain_id)
                rdkit_atom.SetIntProp("chain_idx", chain_idx)
                rdkit_atom.SetIntProp("global_residue_idx", global_res_idx)
                
                idx = mol.AddAtom(rdkit_atom)
                
                # Set position
                if atom_prim.shape is not None:
                    pos = atom_prim.shape.centroid
                else:
                    pos = default_atom_position
                conf.SetAtomPosition(idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
                
                # Store mappings
                atom_handle_to_rdkit_idx[(chain_idx, global_res_idx, atom_handle)] = idx
                atom_map_for_residue[atom_handle] = idx
                
                rdkit_atom_idx += 1
            
            residue_atom_maps.append(atom_map_for_residue)
            global_res_idx += 1
    
    print(f"Added {rdkit_atom_idx} atoms to RDKit molecule")
    
    # ========================================
    # PASS 2: Add INTRA-RESIDUE bonds
    # ========================================
    intra_bond_count = 0
    global_res_idx = 0
    
    for chain_idx, chain in enumerate(univprim.children):
        for residue in chain.children:
            atom_map = residue_atom_maps[global_res_idx]
            
            # Get bonds from internal_connections at residue level
            if hasattr(residue, 'internal_connections') and residue.internal_connections:
                for conn_ref1, conn_ref2 in residue.internal_connections:
                    atom_handle1 = conn_ref1.primitive_handle
                    atom_handle2 = conn_ref2.primitive_handle
                    
                    if atom_handle1 in atom_map and atom_handle2 in atom_map:
                        rdkit_idx1 = atom_map[atom_handle1]
                        rdkit_idx2 = atom_map[atom_handle2]
                        
                        # Get bond type from connector
                        atom1 = residue.fetch_child(atom_handle1)
                        if conn_ref1.connector_handle in atom1.connectors:
                            connector = atom1.connectors[conn_ref1.connector_handle]
                            bond_type = connector.bondtype
                        else:
                            bond_type = Chem.BondType.SINGLE
                        
                        # Check if bond already exists
                        if mol.GetBondBetweenAtoms(rdkit_idx1, rdkit_idx2) is None:
                            mol.AddBond(rdkit_idx1, rdkit_idx2, bond_type)
                            intra_bond_count += 1
            
            global_res_idx += 1
    
    print(f"Added {intra_bond_count} intra-residue bonds")
    
    # ========================================
    # PASS 3: Add INTER-RESIDUE bonds
    # ========================================
    inter_bond_count = 0
    
    for chain_idx, chain in enumerate(univprim.children):
        # Build mapping from residue handle -> global residue index for this chain
        res_handle_to_global: Dict[PrimitiveHandle, int] = {}
        global_res_offset = sum(len(univprim.children[c].children) for c in range(chain_idx))
        
        for local_res_idx, res_handle in enumerate(chain.children_by_handle.keys()):
            res_handle_to_global[res_handle] = global_res_offset + local_res_idx
        
        if hasattr(chain, 'internal_connections') and chain.internal_connections:
            for conn_ref1, conn_ref2 in chain.internal_connections:
                res1_handle = conn_ref1.primitive_handle
                res2_handle = conn_ref2.primitive_handle
                
                try:
                    residue1 = chain.fetch_child(res1_handle)
                    residue2 = chain.fetch_child(res2_handle)
                except Exception:
                    continue
                
                # Get atom handles via external_connectors
                if conn_ref1.connector_handle not in residue1.external_connectors:
                    continue
                if conn_ref2.connector_handle not in residue2.external_connectors:
                    continue
                
                atom_ref1 = residue1.external_connectors[conn_ref1.connector_handle]
                atom_ref2 = residue2.external_connectors[conn_ref2.connector_handle]
                
                atom1_handle = atom_ref1.primitive_handle
                atom2_handle = atom_ref2.primitive_handle
                
                global_res1_idx = res_handle_to_global[res1_handle]
                global_res2_idx = res_handle_to_global[res2_handle]
                
                if atom1_handle not in residue_atom_maps[global_res1_idx]:
                    continue
                if atom2_handle not in residue_atom_maps[global_res2_idx]:
                    continue
                
                rdkit_idx1 = residue_atom_maps[global_res1_idx][atom1_handle]
                rdkit_idx2 = residue_atom_maps[global_res2_idx][atom2_handle]
                
                # Get bond type from connector on residue1
                if conn_ref1.connector_handle in residue1.connectors:
                    connector = residue1.connectors[conn_ref1.connector_handle]
                    bond_type = connector.bondtype
                else:
                    bond_type = Chem.BondType.SINGLE
                
                # Check if bond already exists
                if mol.GetBondBetweenAtoms(rdkit_idx1, rdkit_idx2) is None:
                    mol.AddBond(rdkit_idx1, rdkit_idx2, bond_type)
                    inter_bond_count += 1
    
    print(f"Added {inter_bond_count} inter-residue bonds")
    
    # Add conformer
    mol.AddConformer(conf, assignId=True)
    
    # Add metadata as molecule properties
    mol.SetProp("origin", TOOLKIT_NAME)
    mol.SetProp("chain_labels", ";".join(chain_labels))
    mol.SetProp("n_chains", str(len(chain_labels)))
    mol.SetProp("n_residues_per_chain", ";".join(str(n) for n in residues_per_chain))
    
    if resname_map:
        mol.SetProp("resname_mapping", str(resname_map))
    
    if univprim.label:
        mol.SetProp("_Name", univprim.label)
    
    # Freeze to immutable Mol
    final_mol = mol.GetMol()
    
    return final_mol

"""
# ========================================
# Use the hierarchical export function
# ========================================
print("Converting Primitive hierarchy to RDKit molecule (no flattening)...")
rdkit_mol = primitive_to_rdkit_hierarchical(univprim, resname_map=resname_mapper)

print(f"\nRDKit molecule: {rdkit_mol.GetNumAtoms()} atoms, {rdkit_mol.GetNumBonds()} bonds")

# Show bond type distribution to verify bond orders are preserved
bond_type_counts = Counter(bond.GetBondType().name for bond in rdkit_mol.GetBonds())
print(f"Bond type distribution: {dict(bond_type_counts)}")

# Check for radicals (unsatisfied valences)
radical_count = sum(atom.GetNumRadicalElectrons() for atom in rdkit_mol.GetAtoms())
print(f"Atoms with radical electrons: {radical_count}")

# Write to SDF
save_dir = Path('mupt-built_systems')
save_dir.mkdir(exist_ok=True)
save_path = save_dir / f'ellipsoidal_backmap_{n_chains}x[{chain_len_min}-{chain_len_max}]-mer_chains.sdf'

# Use V3000 format for large molecules (>999 atoms)
writer = SDWriter(str(save_path))
if rdkit_mol.GetNumAtoms() > 999:
    writer.SetForceV3000(True)
writer.write(rdkit_mol)
writer.close()

print(f'\nExported {rdkit_mol.GetNumAtoms()}-atom system to {save_path}')

# Store for later use
mda_rdkit_export = rdkit_mol
display(rdkit_mol)
"""