'''For providing support and validation for chemical line notations'''

from typing import Union
from rdkit import Chem


# CUSTOM TYPEHINTS
type Smiles = str # these are just aliases for now
type Smarts = str # these are just aliases for now
SmilesLike = Union[Smiles, Smarts]

# BOND PRIMITIVES AND RELATED OBJECTS
BOND_PRIMITIVES = '~-=#$:'
BOND_PRIMITIVES_FOR_REGEX = r'[~\-=#$:]' # any of the SMARTS bond primitive chars, with a space to differentiate single-bond hyphen for the regex range char
BOND_INITIALIZERS = {
    'SMILES' : (Chem.Bond     , Chem.BondFromSmiles),
    'SMARTS' : (Chem.QueryBond, Chem.BondFromSmarts),
}

# VALIDATION
def is_valid_SMILES(smiles : Smiles) -> bool:
    '''Check if SMARTS string is valid (according to RDKit)'''
    return (Chem.MolFromSmiles(smiles) is not None)

def is_valid_SMARTS(smarts : Smarts) -> bool:
    '''Check if SMARTS string is valid (according to RDKit)'''
    return (Chem.MolFromSmarts(smarts) is not None)

# UPCONVERSION
def make_chemically_explicit(smiles : Smiles) -> Smiles:
    '''Insert all hydrogens, bond indicators, formal charges and 
    other chemical info implicit in a "bare" SMILES string'''
    ...