'''Readers, writers, and parsers for reading molecules structures into and exporting molecular structures out of the MuPT molecular representation'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from rdkit.Chem.rdmolfiles import SmilesParserParams, SmilesWriteParams

# Module-wide defaults for SMILES parsing
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