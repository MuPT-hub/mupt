'''Test that no information is lost when converting from and then back to RDKit Mols'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

import pytest
from rdkit.Chem.rdchem import Atom, AtomPDBResidueInfo, Mol, RWMol
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs

from mupt.mupr.connection import TraversalDirection
from mupt.mupr.primitives import Primitive
from mupt.roles import PrimitiveRole
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.interfaces.rdkit import importers, primitive_to_rdkit_mols

# TODO: test chemical info (e.g. charge, isotope, etc.) is preserved on atoms

@pytest.fixture(scope='function')
def mol() -> Mol:
    '''A simple test molecule with nontrivial chemical features'''
    rdmol = MolFromSmiles('[NH3+]Cc1c(C#N)c(C-[*:34])ccc1C(-[O-])=O')
    rdmol = AddHs(rdmol)
    # conf_id = EmbedMolecule(mol)

    return rdmol

@pytest.fixture(scope='function')
def primitive(mol : Mol) -> Primitive:
    return importers.primitive_from_rdkit(mol)


def test_valences_permissible(primitive : Primitive) -> None:
    '''Check that chemical valences for all atomic Primitives are among those allowable for their assigned element'''
    assert all(atomprim.is_atom for atomprim in primitive.leaves)


def test_primitive_from_rdkit_defaults_to_saamr_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.role == PrimitiveRole.UNIVERSE
    assert len(primitive.children) == 1
    segment = primitive.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 1
    residue = segment.children[0]
    assert residue.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in residue.children)


def test_primitive_from_rdkit_preserves_legacy_denest_default(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol)

    assert primitive.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)


def test_primitive_from_rdkit_legacy_denest_uses_residue_role(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(
        mol,
        denest=True,
        role=PrimitiveRole.UNIVERSE,
        residue_role=PrimitiveRole.SEGMENT,
    )

    assert primitive.role == PrimitiveRole.SEGMENT


def test_primitive_from_rdkit_denest_false_returns_saamr_hierarchy(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.role == PrimitiveRole.UNIVERSE
    assert len(primitive.children) == 1
    segment = primitive.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 1
    residue = segment.children[0]
    assert residue.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in residue.children)


def test_primitive_from_rdkit_denest_false_preserves_root_metadata() -> None:
    mol = MolFromSmiles("CC.CC")
    mol.SetProp("mupt_test_root", "preserved")

    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.metadata["mupt_test_root"] == "preserved"
    assert primitive.role == PrimitiveRole.UNIVERSE
    assert len(primitive.children) == 2


def test_primitive_from_rdkit_does_not_reclassify_external_prefixed_metadata() -> None:
    mol = MolFromSmiles("CC")
    mol.SetProp("mupt_root_metadata_external", "segment")

    primitive = importers.primitive_from_rdkit(mol, denest=False)

    assert primitive.metadata["mupt_root_metadata_external"] == "segment"
    assert "external" not in primitive.metadata
    assert primitive.children[0].metadata["mupt_root_metadata_external"] == "segment"


def test_primitive_from_rdkit_fallback_residue_key_includes_chain_id() -> None:
    editable = RWMol()
    for chain_id in ("A", "B"):
        atom = Atom(6)
        atom.SetMonomerInfo(
            AtomPDBResidueInfo(
                atomName=" C  ",
                residueName="RES",
                residueNumber=1,
                chainId=chain_id,
            )
        )
        editable.AddAtom(atom)
    editable.AddBond(0, 1)

    primitive = importers.primitive_from_rdkit(Mol(editable), denest=False)

    assert len(primitive.children[0].children) == 2


def test_primitive_from_rdkit_fallback_residue_key_includes_insertion_code() -> None:
    editable = RWMol()
    for insertion_code in ("A", "B"):
        atom = Atom(6)
        pdb_info = AtomPDBResidueInfo(
            atomName=" C  ",
            residueName="RES",
            residueNumber=1,
            chainId="A",
        )
        pdb_info.SetInsertionCode(insertion_code)
        atom.SetMonomerInfo(pdb_info)
        editable.AddAtom(atom)
    editable.AddBond(0, 1)

    primitive = importers.primitive_from_rdkit(Mol(editable), denest=False)

    assert len(primitive.children[0].children) == 2

    exported = primitive_to_rdkit_mols(primitive, {"RES": "RES"})[0]
    insertion_codes = {
        atom.GetPDBResidueInfo().GetInsertionCode()
        for atom in exported.GetAtoms()
    }
    reimported = importers.primitive_from_rdkit(exported, denest=False)
    reimported_insertion_codes = {
        residue.metadata["pdb_insertion_code"]
        for residue in reimported.children[0].children
    }

    assert insertion_codes == {"A", "B"}
    assert reimported_insertion_codes == {"A", "B"}


def test_primitive_from_rdkit_denest_false_preserves_linker_connectors() -> None:
    mol = MolFromSmiles("*-[C:1]-[C:2]-*")

    primitive = importers.primitive_from_rdkit(mol, denest=False)
    segment = primitive.children[0]
    residue = segment.children[0]

    assert len(residue.external_connectors) == 2
    assert len(segment.external_connectors) == 2
    residue_directions = {
        direction
        for conn_handle in residue.external_connectors
        for direction in residue.fetch_connector(conn_handle).anchor.attachables
        if isinstance(direction, TraversalDirection)
    }
    segment_directions = {
        direction
        for conn_handle in segment.external_connectors
        for direction in segment.fetch_connector(conn_handle).anchor.attachables
        if isinstance(direction, TraversalDirection)
    }

    assert residue_directions == {TraversalDirection(1), TraversalDirection(2)}
    assert segment_directions == {TraversalDirection(1), TraversalDirection(2)}


def test_primitive_from_smiles_defaults_to_residue_particle_roles() -> None:
    primitive = primitive_from_smiles("CC", ensure_explicit_Hs=True)

    assert primitive.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)


def test_primitive_from_rdkit_accepts_explicit_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(
        mol,
        denest=False,
        role=PrimitiveRole.UNIVERSE,
        residue_role=PrimitiveRole.RESIDUE,
        atom_role=PrimitiveRole.PARTICLE,
    )

    assert primitive.role == PrimitiveRole.UNIVERSE
    assert primitive.children[0].role == PrimitiveRole.SEGMENT
    assert primitive.children[0].children[0].role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.leaves)


def test_rdkit_export_import_preserves_saamr_hierarchy(
    single_polyethylene_2mer,
    polyethylene_resname_map,
) -> None:
    rdkit_mol = primitive_to_rdkit_mols(single_polyethylene_2mer, polyethylene_resname_map)[0]
    reconstructed = importers.primitive_from_rdkit(rdkit_mol, denest=False)

    assert reconstructed.role == PrimitiveRole.UNIVERSE
    assert len(reconstructed.children) == 1
    segment = reconstructed.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 2
    assert [residue.label for residue in segment.children] == ["head", "tail"]
    assert all(residue.role == PrimitiveRole.RESIDUE for residue in segment.children)
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in reconstructed.leaves)
    assert len(reconstructed.leaves) == len(single_polyethylene_2mer.leaves)
