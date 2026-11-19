"""Writers which convert the MuPT molecular representation out to RDKit Mols"""

from typing import Iterator, Optional
import warnings

import numpy as np
from anytree import PreOrderIter

from rdkit import Chem
from rdkit.Chem.rdchem import (
    Atom,
    BondType,
    ChiralType,
    Mol,
    RWMol,
    Conformer,
    AtomPDBResidueInfo,
)
from rdkit.Geometry import Point3D

from ... import TOOLKIT_NAME
from ...chemistry.rdkit.rdprops import RDPropType, assign_property_to_rdobj
from ...chemistry.rdkit.labelling import RDMOL_NAME_WRITE_PROP
from ...chemistry.conversion import element_to_rdkit_atom

from ...geometry.arraytypes import Vector3
from ...mupr.connection import Connector
from ...mupr.primitives import AtomicPrimitive, Primitive, PrimitiveHandle

from .components import BOND_RANK_KEY, CHIRAL_TAG_KEY, STEREO_METADATA_KEYS
from .strategies import (
    AllAtomRDKitExportStrategy,
    RDKitExportStrategy,
    RDKitMolData,
)


def rdkit_atom_from_atomic_primitive(atomic_primitive: Primitive) -> Atom:
    """Convert an atomic Primitive to an RDKit Atom"""
    if not isinstance(atomic_primitive, AtomicPrimitive):
        raise ValueError("Cannot export non-atomic Primitive to RDKit Atom")

    atom = element_to_rdkit_atom(atomic_primitive.element)
    for key, value in atomic_primitive.metadata.items():
        if key in STEREO_METADATA_KEYS:
            continue  # restored as chiral tags, not copied as props
        assign_property_to_rdobj(atom, key, value, preserve_type=True)

    return atom


def _permutation_is_odd(sequence: list[int]) -> bool:
    """Whether reordering the distinct values of a sequence into sorted order takes an odd number of swaps"""
    n_inversions = sum(
        1
        for i in range(len(sequence))
        for j in range(i + 1, len(sequence))
        if sequence[i] > sequence[j]
    )
    return bool(n_inversions % 2)


def _restore_tetrahedral_stereo(
    mol: RWMol,
    atoms: list[AtomicPrimitive],
    atom_idxs: dict[int, int],
    bond_ranks: dict[tuple[int, int], int],
) -> None:
    """
    Reassign chiral tags recorded on import, correcting for the new bond ordering

    A chiral tag is defined relative to the order of an atom's bonds; the exported Mol
    orders them differently (and ports now bond to other units), so the tag is inverted
    whenever the new order is an odd permutation of the order recorded on import
    """
    for atom in atoms:
        if (tag := atom.metadata.get(CHIRAL_TAG_KEY)) is None:
            continue

        idx = atom_idxs[id(atom)]
        rd_atom = mol.GetAtomWithIdx(idx)
        ranks = [
            bond_ranks.get((idx, bond.GetOtherAtomIdx(idx))) for bond in rd_atom.GetBonds()
        ]
        if None in ranks:  # some bond lacks import ordering; can't restore safely
            continue

        tag = ChiralType(tag)
        if _permutation_is_odd(ranks):
            tag = (
                ChiralType.CHI_TETRAHEDRAL_CCW
                if (tag == ChiralType.CHI_TETRAHEDRAL_CW)
                else ChiralType.CHI_TETRAHEDRAL_CW
            )
        rd_atom.SetChiralTag(tag)


def _add_bond(
    mol: RWMol,
    idx1: int,
    idx2: int,
    bondtype: BondType,
    metadata: dict,
) -> None:
    """Add a bond (with metadata as props) to a writable Mol, flagging aromaticity"""
    if mol.GetBondBetweenAtoms(idx1, idx2) is not None:
        raise ValueError(
            f"Atoms {idx1} and {idx2} are joined by more than one connection, "
            "which can't be represented as an RDKit bond"
        )
    mol.AddBond(idx1, idx2, order=bondtype)
    bond = mol.GetBondBetweenAtoms(idx1, idx2)
    if bondtype == BondType.AROMATIC:
        bond.SetIsAromatic(True)
        mol.GetAtomWithIdx(idx1).SetIsAromatic(True)
        mol.GetAtomWithIdx(idx2).SetIsAromatic(True)

    for key, value in metadata.items():
        if key in STEREO_METADATA_KEYS:
            continue
        assign_property_to_rdobj(bond, key, value, preserve_type=True)


def rdkit_mol_from_primitive(
    primitive: Primitive,
    default_atom_position: Optional[Vector3] = None,
) -> Mol:
    """
    Convert a Primitive hierarchy whose leaves are atomic Primitives to a single RDKit Mol

    Free Connectors become "*" linker atoms (carrying the Connector label as their
    map number, if an int); atom positions are written to a Conformer, and tetrahedral
    stereo recorded on import is restored
    """
    if default_atom_position is None:
        default_atom_position = np.zeros(3, dtype=float)

    atoms: list[AtomicPrimitive] = [
        prim for prim in PreOrderIter(primitive) if isinstance(prim, AtomicPrimitive)
    ]
    if not atoms:
        raise ValueError(f"{primitive!r} contains no atomic Primitives to export")

    mol = RWMol()
    positions: list[np.ndarray] = []
    atom_idxs: dict[int, int] = {}  # id(atomic Primitive) -> RDKit atom index
    for atom in atoms:
        atom_idxs[id(atom)] = mol.AddAtom(rdkit_atom_from_atomic_primitive(atom))
        positions.append(
            atom.shape.centroid if (atom.shape is not None) else default_atom_position
        )

    # (atom index, neighbor index) -> rank of the atom's Connector along that bond
    bond_ranks: dict[tuple[int, int], int] = {}
    for atom in atoms:
        idx = atom_idxs[id(atom)]
        for conn in atom.connections.connectors:
            neighbor = conn.neighbor.holder if conn.has_neighbor else None
            if (neighbor is not None) and (id(neighbor) in atom_idxs):
                nb_idx = atom_idxs[id(neighbor)]
                if idx < nb_idx:  # each bond is seen from both ends; add once
                    _add_bond(
                        mol,
                        idx,
                        nb_idx,
                        conn.bondtype,
                        {**conn.metadata, **conn.neighbor.metadata},
                    )
            else:  # free (or leads outside the exported hierarchy) -> linker atom
                linker = Atom(0)
                if isinstance(conn.label, int) and not isinstance(conn.label, bool):
                    linker.SetAtomMapNum(conn.label)
                nb_idx = mol.AddAtom(linker)
                positions.append(conn.linker.position)
                _add_bond(mol, idx, nb_idx, conn.bondtype, conn.metadata)

            if BOND_RANK_KEY in conn.metadata:
                bond_ranks[(idx, nb_idx)] = conn.metadata[BOND_RANK_KEY]

    _restore_tetrahedral_stereo(mol, atoms, atom_idxs, bond_ranks)

    conf = Conformer(mol.GetNumAtoms())
    for idx, position in enumerate(positions):
        conf.SetAtomPosition(idx, Point3D(*map(float, position)))
    mol.AddConformer(conf, assignId=True)

    assign_property_to_rdobj(mol, "origin", TOOLKIT_NAME, preserve_type=True)
    for key, value in primitive.metadata.items():
        assign_property_to_rdobj(mol, key, value, preserve_type=True)

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # still return a usable Mol for partially-specified chemistry
        mol.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    if primitive.label is not None:
        mol.SetProp(RDMOL_NAME_WRITE_PROP, str(primitive.label))

    return mol


def primitive_to_rdkit(
    primitive: Primitive,
    default_atom_position: Optional[Vector3] = None,
) -> Mol:
    """
    Convert a Primitive hierarchy to a single RDKit Mol (see rdkit_mol_from_primitive)

    Returns a single Mol instance, even if the Primitive
    represents a collection of multiple disconnected molecules
    """
    warnings.warn(
        "To skip flattening step; use the role-aware "
        "primitive_to_rdkit_mols() exporter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rdkit_mol_from_primitive(primitive, default_atom_position=default_atom_position)


# TB: these should be moved to a dedicated .pdb module, NOT live in RDKit core
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


def _apply_rdkit_mol_metadata(
    mol: RWMol, data: RDKitMolData, root_metadata: dict
) -> None:
    """Attach root and segment metadata to one RDKit Mol."""
    assign_property_to_rdobj(mol, "origin", TOOLKIT_NAME, preserve_type=True)
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
    default_atom_position: Optional[Vector3] = None,
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
        strategy = AllAtomRDKitExportStrategy(
            default_atom_position=default_atom_position
        )

    first_residue_idx = 0
    for segment_idx, data in enumerate(
        strategy.iter_mol_data(primitive, resname_map=resname_map)
    ):
        yield _mol_from_rdkit_data(
            data,
            segment_idx,
            first_residue_idx,
            root_metadata=primitive.metadata,
        )
        first_residue_idx += max(data.atom_resids, default=0)
