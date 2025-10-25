'''Unit tests for Mol saniitization wrappers and utilities'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest

from typing import Union, Optional
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.rdmolops import AromaticityModel, AROMATICITY_RDKIT, AROMATICITY_MDL

from mupt.chemistry.sanitization import sanitized_mol


# HELPER FUNCTIONS
def get_rdatom_with_mapnum(
    mol : Mol,
    mapnum : int,
    as_index : bool=True,
) -> Optional[Union[int, Mol]]:
    '''
    Get the RDKit atom index of the atom with the specified atom map number in the given molecule.
    
    Parameters
    ----------
    mol : Mol
        The RDKit Mol to search
    mapnum : int
        The atom map number to search for
        
    Returns
    -------
    atom_idx : Optional[int]
        The RDKit atom index of the atom with the specified atom map number, or None if not found
    '''
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == mapnum:
            return atom.GetIdx() if as_index else atom
    else:
        return None


# TESTS
@pytest.mark.parametrize(
    'smiles,arom_model,expected_valence',
    [
        ('c1[nH:1]c2ccccc2c1C[C@H](N)C(=O)O', AROMATICITY_RDKIT, 4), # tryptophan (targetting carboxyl oxygen) - +1 net valence 
        ('c1[nH:1]c2ccccc2c1C[C@H](N)C(=O)O', AROMATICITY_MDL, 3),   # tryptophan (targetting carboxyl oxygen) - expected electronically-consistent valence
        ('c12C(=O)[O:1]C(=O)c2cc3C(=O)OC(=O)c3c1', AROMATICITY_RDKIT, 3),  # pmda (targetting amide nitrogen) - +1 net valence 
        ('c12C(=O)[O:1]C(=O)c2cc3C(=O)OC(=O)c3c1', AROMATICITY_MDL, 2),    # pmda (targetting amide nitrogen) - expected electronically-consistent valence
    ]
)
def test_ringed_system_aromaticity_by_model(
    smiles : str,
    arom_model : AromaticityModel,
    expected_valence : int,
    targ_atom_map_num : int=1
) -> None:
    '''
    Test that the choice of aromaticity model yields different aromaticity assignments on known pathological molecules
    
    Primarily intended as sanity check that AROMATICITY_MDL assigns bond orders
    consistent with atomic valence, while AROMATICITY_RDKIT sometimes does not.
    '''
    mol = MolFromSmiles(smiles)
    cleanmol = sanitized_mol(mol, add_Hs=True, aromaticity_model=arom_model)
    targ_atom = get_rdatom_with_mapnum(cleanmol, targ_atom_map_num, as_index=False)
    targ_valence = round(sum(bond.GetBondTypeAsDouble() for bond in targ_atom.GetBonds()))
    
    assert targ_valence == expected_valence
    

