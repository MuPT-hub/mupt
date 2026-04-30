'''Test that no information is lost when converting from and then back to RDKit Mols'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

import pytest
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs

from mupt.chemistry.core import valence_allowed
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
    assert all( # DEV: break off into parameterized test for individual atomic Primitive?
        valence_allowed(atomprim.element.number, atomprim.element.charge, atomprim.valence)
            for atomprim in primitive.leaves
    )


def test_primitive_from_rdkit_defaults_to_saamr_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol)

    assert primitive.role == PrimitiveRole.UNIVERSE
    assert len(primitive.children) == 1
    segment = primitive.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 1
    residue = segment.children[0]
    assert residue.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in residue.children)


def test_primitive_from_smiles_defaults_to_residue_particle_roles() -> None:
    primitive = primitive_from_smiles("CC", ensure_explicit_Hs=True)

    assert primitive.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)


def test_primitive_from_rdkit_accepts_explicit_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(
        mol,
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
    reconstructed = importers.primitive_from_rdkit(rdkit_mol)

    assert reconstructed.role == PrimitiveRole.UNIVERSE
    assert len(reconstructed.children) == 1
    segment = reconstructed.children[0]
    assert segment.role == PrimitiveRole.SEGMENT
    assert len(segment.children) == 2
    assert [residue.label for residue in segment.children] == ["head", "tail"]
    assert all(residue.role == PrimitiveRole.RESIDUE for residue in segment.children)
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in reconstructed.leaves)
    assert len(reconstructed.leaves) == len(single_polyethylene_2mer.leaves)
