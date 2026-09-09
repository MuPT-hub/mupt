'''For providing support and validation for chemical line notations'''
# DEV: resist the temptation to merge this into interfaces.smiles; will cause a circular import own the line
# Namely, interfaces.rdkit depends on utils here, and importers/exports in interfaces.smileslib depend in turn on those RDKit utils

from typing import Union
from rdkit import Chem

from rdkit.Chem.rdmolfiles import SmilesParserParams, SmilesWriteParams


# LIBRARY-WIDE DEFAULTS FOR SMILES I/O
## Reading
DEFAULT_SMILES_READ_PARAMS = SmilesParserParams()
DEFAULT_SMILES_READ_PARAMS.sanitize = False
DEFAULT_SMILES_READ_PARAMS.removeHs = False
DEFAULT_SMILES_READ_PARAMS.allowCXSMILES = True

## Writing
DEFAULT_SMILES_WRITE_PARAMS = SmilesWriteParams()
DEFAULT_SMILES_WRITE_PARAMS.doIsomericSmiles = True
DEFAULT_SMILES_WRITE_PARAMS.doKekule         = False
DEFAULT_SMILES_WRITE_PARAMS.canonical        = True
DEFAULT_SMILES_WRITE_PARAMS.allHsExplicit    = False
DEFAULT_SMILES_WRITE_PARAMS.doRandom         = False

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