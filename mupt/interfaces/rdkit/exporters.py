'''Writers which convert the MuPT molecular representation out to RDKit Mols'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from anytree import PreOrderIter
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
from ...roles import PrimitiveRole
from .._shared.topology import _pdb_resname, _resolve_to_atom


@dataclass
class RDKitMolData:
    """Container for one segment's RDKit-exportable topology data."""

    segment: Primitive
    atoms: list[Primitive] = field(default_factory=list)
    atom_positions: list[np.ndarray] = field(default_factory=list)
    atom_resnames: list[str] = field(default_factory=list)
    atom_resids: list[int] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_refs: list[tuple[Primitive, object]] = field(default_factory=list)


class RDKitExportStrategy(ABC):
    """Abstract strategy for collecting RDKit-exportable topology data."""

    @abstractmethod
    def validate(self, root: Primitive) -> None:
        """Validate role assignment and hierarchy preconditions for export."""

    @abstractmethod
    def collect_mols(self, root: Primitive, resname_map: dict[str, str]) -> list[RDKitMolData]:
        """Collect one topology dataset per RDKit Mol to build."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name for this strategy."""


class AllAtomRDKitExportStrategy(RDKitExportStrategy):
    """Role-aware all-atom RDKit export strategy."""

    def __init__(self, default_atom_position: Optional[np.ndarray] = None) -> None:
        if default_atom_position is None:
            self.default_atom_position = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            default_atom_position = np.asarray(default_atom_position, dtype=float)
            if default_atom_position.shape != (3,):
                raise ValueError('default_atom_position must be a 3-dimensional vector')
            self.default_atom_position = default_atom_position

    @property
    def label(self) -> str:
        """Human-readable strategy name."""
        return "All-atom"

    def validate(self, root: Primitive) -> None:
        """Validate role assignments needed for all-atom RDKit export."""
        if root.role != PrimitiveRole.UNIVERSE:
            raise ValueError(
                "Root Primitive must have role=PrimitiveRole.UNIVERSE. "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )

        segment_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]
        residue_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.RESIDUE]
        if not segment_nodes:
            raise ValueError("No SEGMENT-role Primitives found in hierarchy.")
        if not residue_nodes:
            raise ValueError("No RESIDUE-role Primitives found in hierarchy.")

        for leaf in root.leaves:
            if leaf.role != PrimitiveRole.PARTICLE:
                raise ValueError("All leaves must have role=PrimitiveRole.PARTICLE.")
            if leaf.element is None:
                raise ValueError(
                    f"Leaf Primitive '{leaf}' has role=PARTICLE but no element assigned. "
                    "AllAtomRDKitExportStrategy requires atomic PARTICLE leaves."
                )

        for seg in segment_nodes:
            nested_segs = [
                node for node in PreOrderIter(seg)
                if node.role == PrimitiveRole.SEGMENT and node is not seg
            ]
            if nested_segs:
                raise ValueError(
                    f"SEGMENT '{seg.label}' contains nested SEGMENT(s) "
                    f"{[n.label for n in nested_segs]}."
                )

        for res in residue_nodes:
            nested_res = [
                node for node in PreOrderIter(res)
                if node.role == PrimitiveRole.RESIDUE and node is not res
            ]
            if nested_res:
                raise ValueError(
                    f"RESIDUE '{res.label}' contains nested RESIDUE(s) "
                    f"{[n.label for n in nested_res]}."
                )

    def collect_mols(self, root: Primitive, resname_map: dict[str, str]) -> list[RDKitMolData]:
        """Walk the hierarchy and collect one RDKit topology per segment."""
        self.validate(root)

        mols_data: list[RDKitMolData] = []
        for segment in [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]:
            data = RDKitMolData(segment=segment)
            atom_id_to_local: dict[int, int] = {}
            mapped_atom_ids: set[int] = set()
            resid_counter = 1

            for residue in [node for node in PreOrderIter(segment) if node.role == PrimitiveRole.RESIDUE]:
                resname = _pdb_resname(residue.label, resname_map)
                particles = [
                    node for node in PreOrderIter(residue)
                    if node.is_leaf and node.role == PrimitiveRole.PARTICLE
                ]
                for atom in particles:
                    atom_id_to_local[id(atom)] = len(data.atoms)
                    mapped_atom_ids.add(id(atom))
                    data.atoms.append(atom)
                    if atom.shape is not None:
                        data.atom_positions.append(atom.shape.centroid)
                    else:
                        data.atom_positions.append(self.default_atom_position)
                    data.atom_resnames.append(resname)
                    data.atom_resids.append(resid_counter)
                resid_counter += 1

            segment_particles = [
                node for node in PreOrderIter(segment)
                if node.is_leaf and node.role == PrimitiveRole.PARTICLE
            ]
            if {id(atom) for atom in segment_particles} != mapped_atom_ids:
                raise ValueError(
                    f"SEGMENT '{segment.label}' contains PARTICLE leaves that are not "
                    "mapped through RESIDUE-role nodes."
                )

            bonds_set: set[tuple[int, int]] = set()
            for node in PreOrderIter(segment):
                if node.is_leaf or not node.internal_connections:
                    continue

                for conn_ref_pair in node.internal_connections:
                    conn_ref1, conn_ref2 = sorted(
                        conn_ref_pair,
                        key=lambda cr: (cr.primitive_handle, cr.connector_handle),
                    )
                    atom1 = _resolve_to_atom(node, conn_ref1)
                    atom2 = _resolve_to_atom(node, conn_ref2)
                    idx1 = atom_id_to_local[id(atom1)]
                    idx2 = atom_id_to_local[id(atom2)]
                    bond_pair = tuple(sorted((idx1, idx2)))
                    if bond_pair in bonds_set:
                        continue

                    data.bonds.append(bond_pair)
                    data.bond_refs.append((node, conn_ref1))
                    bonds_set.add(bond_pair)

            if data.bonds:
                sorted_bonds = sorted(
                    zip(data.bonds, data.bond_refs), key=lambda pair: pair[0]
                )
                data.bonds = [bond for bond, _ in sorted_bonds]
                data.bond_refs = [bond_ref for _, bond_ref in sorted_bonds]

            mols_data.append(data)

        return mols_data


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
    
