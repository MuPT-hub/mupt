'''Test that no information is lost when converting from and then back to RDKit Mols'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs

from mupt.chemistry.core import valence_allowed
from mupt.mupr.primitives import Primitive
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