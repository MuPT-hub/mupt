'''Testing assignment of labels to RDKit objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest

from rdkit import Chem
from mupt.interfaces.rdkit.labelling import RDMOL_NAME_READ_PROP_PRECEDENCE, name_for_rdkit_mol
from mupt.chemistry.smiles import DEFAULT_SMILES_WRITE_PARAMS


@pytest.fixture(scope='function') # avoid cross-contaminating tests
def mol() -> Chem.Mol:
    '''A simple RDKit Mol for testing'''
    smi = 'c1ccccc1C(=O)O'
    return Chem.MolFromSmiles(smi)

@pytest.mark.parametrize('prop', RDMOL_NAME_READ_PROP_PRECEDENCE)
def test_rdmol_name_from_prop(mol : Chem.Mol, prop : str) -> None:
    '''Test that a Mol with a given name property set is labelled with that property'''
    expected_name : str = f'test_name_for_{prop}' # ensure names are different to guarantee no false-positive from prior name sets
    mol.SetProp(prop, expected_name)
    assert name_for_rdkit_mol(mol) == expected_name

def test_rdmol_name_smiles_fallback(mol : Chem.Mol) -> None:
    '''Test that a Mol with no explicit name set is labelled with its SMILES (according to default writer params)'''
    expected_name : str = Chem.MolToSmiles(mol, params=DEFAULT_SMILES_WRITE_PARAMS)
    assert name_for_rdkit_mol(mol) == expected_name
    
@pytest.mark.skipif(len(RDMOL_NAME_READ_PROP_PRECEDENCE) < 2, reason="Not enough precedence levels to test")
def test_higher_precedence_overrides_lower(mol : Chem.Mol) -> None:
    '''Test that, when multiple name properties are set, the one with the highest precedence is returned by the labeller'''
    primary_prop_attr = RDMOL_NAME_READ_PROP_PRECEDENCE[0]
    secondary_prop_attr = RDMOL_NAME_READ_PROP_PRECEDENCE[-1]
    
    mol.SetProp(primary_prop_attr, 'higher_precedence_name')
    mol.SetProp(secondary_prop_attr, 'lower_precedence_name')
    
    assert name_for_rdkit_mol(mol) == 'higher_precedence_name'
    