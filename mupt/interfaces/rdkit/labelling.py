'''For assigning and deriving labels for RDKit objects (e.g. Mols, Bonds, and Atoms)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem.rdmolfiles import MolToSmiles, SmilesWriteParams
from ...chemistry.smiles import DEFAULT_SMILES_WRITE_PARAMS


RDMOL_NAME_PROP_PRECEDENCE : tuple[str] = ( 
    'name',
    'Name',
    'label',
    'Label',
    '_name',
    '_Name',
    '_label',
    '_Label',
)
RDMOL_NAME_PROP : str = RDMOL_NAME_PROP_PRECEDENCE[0]

def name_for_rdkit_mol(
    mol : Mol,
    smiles_writer_params : SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
) -> str:
    '''
    Fetch a name (as string) for an RDKit mol
    
    Will attempt to fetch from properties set on Mol or, 
    if none are found, falls back to SMILES representation of that Mol
    '''
    for prop in RDMOL_NAME_PROP_PRECEDENCE:
        if mol.HasProp(prop):
            return mol.GetProp(prop)
    else:
        return MolToSmiles(mol, params=smiles_writer_params)