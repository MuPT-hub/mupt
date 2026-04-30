'''Writers which convert the MuPT molecular representation out to RDKit Mols'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'


from typing import Optional

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

from .rdprops import RDPropType, assign_property_to_rdobj
from .labelling import RDMOL_NAME_WRITE_PROP
from ... import TOOLKIT_NAME
from ...geometry.arraytypes import Shape
from ...chemistry.conversion import element_to_rdkit_atom
from ...mupr.connection import Connector
from ...mupr.primitives import Primitive, PrimitiveHandle
from .strategies import AllAtomRDKitExportStrategy, RDKitExportStrategy, RDKitMolData


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


def _chain_id(segment_idx: int) -> str:
    """Return a deterministic PDB chain ID for a segment index."""
    chain_id = ""
    idx = segment_idx
    while True:
        chain_id = chr(ord('A') + (idx % 26)) + chain_id
        idx = idx // 26 - 1
        if idx < 0:
            return chain_id


def _atom_pdb_name(atom: Primitive, atom_idx_in_residue: int) -> str:
    """Return a PDB-width atom name derived from element and residue-local index."""
    atom_name = f"{atom.element.symbol}{atom_idx_in_residue + 1}"
    if len(atom.element.symbol) == 1:
        return f" {atom_name:<3}"
    return f"{atom_name:<4}"


def _mol_from_rdkit_data(data: RDKitMolData, segment_idx: int) -> Mol:
    """Build an RDKit Mol from collected role-aware topology data."""
    mol = RWMol()
    conf = Conformer(len(data.atoms))
    chain_id = _chain_id(segment_idx)
    residue_atom_counts: dict[int, int] = {}

    for atom_idx, atom_prim in enumerate(data.atoms):
        rdkit_atom = rdkit_atom_from_atomic_primitive(atom_prim)
        resid = data.atom_resids[atom_idx]
        atom_idx_in_residue = residue_atom_counts.get(resid, 0)
        residue_atom_counts[resid] = atom_idx_in_residue + 1

        rdkit_atom.SetMonomerInfo(
            AtomPDBResidueInfo(
                atomName=_atom_pdb_name(atom_prim, atom_idx_in_residue),
                serialNumber=atom_idx + 1,
                residueName=data.atom_resnames[atom_idx],
                residueNumber=resid,
                chainId=chain_id,
                isHeteroAtom=True,
            )
        )
        rdkit_atom.SetProp("residue_name", data.atom_resnames[atom_idx])
        rdkit_atom.SetIntProp("residue_id", resid)
        rdkit_atom.SetProp("chain_id", chain_id)

        idx = mol.AddAtom(rdkit_atom)
        pos = data.atom_positions[atom_idx]
        conf.SetAtomPosition(idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))

    for (idx1, idx2), (parent, conn_ref) in zip(data.bonds, data.bond_refs):
        conn = parent.fetch_connector_on_child(conn_ref)
        mol.AddBond(idx1, idx2, order=conn.bondtype)
        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        for bond_key, bond_value in conn.metadata.items():
            assign_property_to_rdobj(bond, bond_key, bond_value, preserve_type=True)

    assign_property_to_rdobj(mol, 'origin', TOOLKIT_NAME, preserve_type=True)
    if data.segment.label is not None:
        mol.SetProp(RDMOL_NAME_WRITE_PROP, str(data.segment.label))
    for key, value in data.segment.metadata.items():
        assign_property_to_rdobj(mol, key, value, preserve_type=True)

    mol.AddConformer(conf, assignId=True)
    return Mol(mol)


def primitive_to_rdkit_mols(
    primitive: Primitive,
    resname_map: dict[str, str],
    default_atom_position: Optional[np.ndarray[Shape[3], float]] = None,
    strategy: Optional[RDKitExportStrategy] = None,
) -> list[Mol]:
    """Convert a role-annotated Primitive hierarchy to one RDKit Mol per segment."""
    if strategy is None:
        strategy = AllAtomRDKitExportStrategy(default_atom_position=default_atom_position)
    mols_data = strategy.collect_mols(primitive, resname_map=resname_map)
    return [
        _mol_from_rdkit_data(data, segment_idx)
        for segment_idx, data in enumerate(mols_data)
    ]
    
