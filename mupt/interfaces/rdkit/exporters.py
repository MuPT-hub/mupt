'''Writers which convert the MuPT molecular representation out to RDKit Mols'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'


from collections.abc import Iterator
from typing import Optional

import numpy as np

from rdkit.Chem.rdchem import (
    Atom,
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

    DEV: This is the legacy flattened exporter. We recommend replacing downstream
    workflows with primitive_to_rdkit_mols() and removing this path after reviewer
    approval, rather than abstracting shared bond or metadata helpers around code
    that is likely to be retired.

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


PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PDB_MAX_RESIDUE_NUMBER = 9999
MUPT_RDKIT_ATOM_PROPS = (
    "chain_id",
    "residue_id",
    "residue_name",
    "mupt_segment_index",
    "mupt_segment_label",
    "mupt_residue_index",
    "mupt_residue_label",
    "mupt_particle_index",
    "mupt_particle_label",
)


def _pdb_chain_and_resid(global_residue_idx: int) -> tuple[str, int]:
    """
    Return PDB-compatible surrogate chain/residue identifiers for export metadata.

    These are not true MuPT segment/chain semantics. Exported segment records are
    assigned sequentially as A:1 through A:9999, then B:1, and so on for
    PDB/OpenFF-style metadata fields with limited chain/residue capacity. MuPT
    provenance is preserved separately on atom props such as
    mupt_segment_index and mupt_residue_index.
    """
    chain_idx, resid_offset = divmod(global_residue_idx, PDB_MAX_RESIDUE_NUMBER)
    if chain_idx >= len(PDB_CHAIN_IDS):
        raise ValueError(
            "Role-aware RDKit export exceeded PDB chain/residue capacity "
            f"({len(PDB_CHAIN_IDS) * PDB_MAX_RESIDUE_NUMBER} residues). "
            "Use a topology format with larger identifier fields."
        )
    return PDB_CHAIN_IDS[chain_idx], resid_offset + 1


def _atom_pdb_name(atom: Primitive, atom_idx_in_residue: int) -> str:
    """Return a PDB-width atom name from element and residue-local index."""
    atom_name = f"{atom.element.symbol}{atom_idx_in_residue + 1}"
    if len(atom.element.symbol) == 1:
        return f" {atom_name:<3}"
    return f"{atom_name:<4}"


def _add_rdkit_atoms(
    mol: RWMol,
    conf: Conformer,
    data: RDKitMolData,
    segment_idx: int,
    first_residue_idx: int,
) -> None:
    """Insert RDKit atoms, positions, and per-atom MuPT metadata."""
    residue_atom_counts: dict[int, int] = {}

    for atom_idx, atom_prim in enumerate(data.atoms):
        rdkit_atom = rdkit_atom_from_atomic_primitive(atom_prim)
        local_resid = data.atom_resids[atom_idx]
        chain_id, resid = _pdb_chain_and_resid(first_residue_idx + local_resid - 1)
        atom_idx_in_residue = residue_atom_counts.get(local_resid, 0)
        residue_atom_counts[local_resid] = atom_idx_in_residue + 1

        pdb_info = AtomPDBResidueInfo(
            atomName=_atom_pdb_name(atom_prim, atom_idx_in_residue),
            serialNumber=atom_idx + 1,
            residueName=data.atom_resnames[atom_idx],
            residueNumber=resid,
            chainId=chain_id,
            isHeteroAtom=True,
        )
        if data.atom_insertion_codes[atom_idx]:
            pdb_info.SetInsertionCode(data.atom_insertion_codes[atom_idx])
        rdkit_atom.SetMonomerInfo(pdb_info)
        rdkit_atom.SetProp("residue_name", data.atom_resnames[atom_idx])
        rdkit_atom.SetIntProp("residue_id", resid)
        rdkit_atom.SetProp("chain_id", chain_id)
        rdkit_atom.SetIntProp("mupt_segment_index", segment_idx)
        rdkit_atom.SetProp("mupt_segment_label", str(data.segment.label))
        rdkit_atom.SetIntProp("mupt_residue_index", local_resid)
        rdkit_atom.SetProp("mupt_residue_label", data.atom_residue_labels[atom_idx])
        rdkit_atom.SetIntProp("mupt_particle_index", atom_idx)
        rdkit_atom.SetProp("mupt_particle_label", data.atom_particle_labels[atom_idx])

        idx = mol.AddAtom(rdkit_atom)
        pos = data.atom_positions[atom_idx]
        conf.SetAtomPosition(idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))


def _add_rdkit_bonds(mol: RWMol, data: RDKitMolData) -> None:
    """Insert internal bonds and merged bond metadata."""
    for (idx1, idx2), (parent, conn_refs) in zip(data.bonds, data.bond_refs):
        conn1 = parent.fetch_connector_on_child(conn_refs[0])
        conn2 = parent.fetch_connector_on_child(conn_refs[1])
        mol.AddBond(idx1, idx2, order=conn1.bondtype)
        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        bond_metadata: dict[str, RDPropType] = {
            **conn1.metadata,
            **conn2.metadata,
        }
        for bond_key, bond_value in bond_metadata.items():
            assign_property_to_rdobj(bond, bond_key, bond_value, preserve_type=True)


def _add_rdkit_linkers(mol: RWMol, conf: Conformer, data: RDKitMolData) -> None:
    """Insert linker atoms and their bond metadata."""
    for atom_idx, parent, conn_ref in data.linker_refs:
        conn = parent.fetch_connector_on_child(conn_ref)
        anchor_atom = mol.GetAtomWithIdx(atom_idx)
        linker_idx = mol.AddAtom(Atom(0))
        anchor_pdb_info = anchor_atom.GetPDBResidueInfo()
        linker_atom = mol.GetAtomWithIdx(linker_idx)
        if anchor_pdb_info is not None:
            pdb_info = AtomPDBResidueInfo(
                atomName=" *  ",
                serialNumber=linker_idx + 1,
                residueName=anchor_pdb_info.GetResidueName(),
                residueNumber=anchor_pdb_info.GetResidueNumber(),
                chainId=anchor_pdb_info.GetChainId(),
                isHeteroAtom=True,
            )
            if anchor_pdb_info.GetInsertionCode():
                pdb_info.SetInsertionCode(anchor_pdb_info.GetInsertionCode())
            linker_atom.SetMonomerInfo(pdb_info)
            linker_atom.SetProp("residue_name", anchor_atom.GetProp("residue_name"))
            linker_atom.SetIntProp("residue_id", anchor_atom.GetIntProp("residue_id"))
            linker_atom.SetProp("chain_id", anchor_atom.GetProp("chain_id"))
        mol.AddBond(atom_idx, linker_idx, order=conn.bondtype)
        conf.SetAtomPosition(
            linker_idx,
            Point3D(
                float(conn.linker.position[0]),
                float(conn.linker.position[1]),
                float(conn.linker.position[2]),
            ),
        )
        bond = mol.GetBondBetweenAtoms(atom_idx, linker_idx)
        for bond_key, bond_value in conn.metadata.items():
            assign_property_to_rdobj(bond, bond_key, bond_value, preserve_type=True)


def _apply_rdkit_mol_metadata(mol: RWMol, data: RDKitMolData, root_metadata: dict) -> None:
    """Attach root and segment metadata to one RDKit Mol."""
    assign_property_to_rdobj(mol, 'origin', TOOLKIT_NAME, preserve_type=True)
    if data.segment.label is not None:
        mol.SetProp(RDMOL_NAME_WRITE_PROP, str(data.segment.label))
    for key, value in root_metadata.items():
        assign_property_to_rdobj(mol, key, value, preserve_type=True)
    for key, value in data.segment.metadata.items():
        assign_property_to_rdobj(mol, key, value, preserve_type=True)


def _mol_from_rdkit_data(
    data: RDKitMolData,
    segment_idx: int,
    first_residue_idx: int,
    root_metadata: dict,
) -> Mol:
    """Build an RDKit Mol from collected role-aware topology data."""
    mol = RWMol()
    conf = Conformer(len(data.atoms) + len(data.linker_refs))
    _add_rdkit_atoms(mol, conf, data, segment_idx, first_residue_idx)
    _add_rdkit_bonds(mol, data)
    _add_rdkit_linkers(mol, conf, data)
    _apply_rdkit_mol_metadata(mol, data, root_metadata)

    mol.AddConformer(conf, assignId=True)
    final_mol = Mol(mol)
    final_mol.UpdatePropertyCache(strict=True)
    return final_mol


def primitive_to_rdkit_mols(
    primitive: Primitive,
    resname_map: dict[str, str],
    default_atom_position: Optional[np.ndarray[Shape[3], float]] = None,
    strategy: Optional[RDKitExportStrategy] = None,
) -> Iterator[Mol]:
    """
    Yield one RDKit Mol per segment from a role-annotated Primitive hierarchy.

    The strategy performs topology validation as part of iteration. For the
    default all-atom strategy, ``iter_mol_data()`` first builds the shared
    SAAMR role topology index and raises if the hierarchy cannot be exported.
    This keeps the exporter on the same EAFP path as other MuPT interfaces and
    avoids a separate preflight traversal.
    """
    if strategy is None:
        strategy = AllAtomRDKitExportStrategy(default_atom_position=default_atom_position)

    first_residue_idx = 0
    for segment_idx, data in enumerate(strategy.iter_mol_data(primitive, resname_map=resname_map)):
        yield _mol_from_rdkit_data(
            data,
            segment_idx,
            first_residue_idx,
            root_metadata=primitive.metadata,
        )
        first_residue_idx += max(data.atom_resids, default=0)
    
