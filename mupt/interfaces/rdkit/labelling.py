'''For assigning and deriving labels for RDKit objects (e.g. Mols, Bonds, and Atoms)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem.rdmolfiles import MolToSmiles, SmilesWriteParams
from ...chemistry.smiles import DEFAULT_SMILES_WRITE_PARAMS


# Static reference for RDKit mol naming
## Following Postel's Law here; many options for valid name to read, but only one prescribed for write
RDMOL_NAME_READ_PROP_PRECEDENCE : tuple[str] = ( 
    '_Name',
    '_name',
    'name',
    'Name',
    '_Label',
    '_label',
    'label',
    'Label',
)
## "magic" property used to write molecule name to CTABs (https://www.rdkit.org/docs/RDKit_Book.html#romol-mol-in-python)
RDMOL_NAME_WRITE_PROP : str = '_Name' 

def name_for_rdkit_mol(
    mol : Mol,
    smiles_writer_params : SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
) -> str:
    '''
    Fetch a name (as string) for an RDKit mol
    
    Will attempt to fetch from properties set on Mol or, 
    if none are found, falls back to SMILES representation of that Mol
    '''
    for prop in RDMOL_NAME_READ_PROP_PRECEDENCE:
        if mol.HasProp(prop):
            return mol.GetProp(prop)
    else:
        return MolToSmiles(mol, params=smiles_writer_params)