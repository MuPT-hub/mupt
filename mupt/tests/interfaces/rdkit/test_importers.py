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
from mupt.interfaces.rdkit import importers

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
            for atomprim in primitive.children
    )


def test_primitive_from_rdkit_defaults_to_residue_particle_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(mol)

    assert primitive.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)


def test_primitive_from_smiles_defaults_to_residue_particle_roles() -> None:
    primitive = primitive_from_smiles("CC", ensure_explicit_Hs=True)

    assert primitive.role == PrimitiveRole.RESIDUE
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)


def test_primitive_from_rdkit_accepts_explicit_roles(mol: Mol) -> None:
    primitive = importers.primitive_from_rdkit(
        mol,
        role=PrimitiveRole.SEGMENT,
        atom_role=PrimitiveRole.PARTICLE,
    )

    assert primitive.role == PrimitiveRole.SEGMENT
    assert all(atom.role == PrimitiveRole.PARTICLE for atom in primitive.children)
