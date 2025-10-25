'''
Wrappers and reference for RDKit Mol sanitization operations, aromaticity handling, and hydrogen addition/removal
Intended to ensure consistent rules are applied when sanitizing within MuPT
'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional, Union

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import (
    AddHs,
    SanitizeMol,
    SanitizeFlags,
    SANITIZE_NONE,
    SANITIZE_ALL,
    Kekulize,
    SetAromaticity,
    AromaticityModel,
    AROMATICITY_MDL,
)


def sanitized_mol(
    mol : Mol,
    add_Hs : bool=False,
    sanitize_ops : Union[None, int, SanitizeFlags]=SANITIZE_ALL,
    aromaticity_model : Optional[AromaticityModel]=AROMATICITY_MDL,
) -> Mol:
    '''
    Return a copy of an RDKit Mol with the specified hydrogen cleanup, sanitization, and aromaticity inference applied
    
    Parameters
    ----------
    mol : Mol
        The RDKit Mol to sanitize
    add_Hs : bool, default=False
        Whether to add hydrogens to the molecule before sanitization
    sanitize_ops : Union[None, int, SanitizeFlags], default=SANITIZE_ALL
        The sanitization operations to perform on the molecule.
        If None, defaults to SANITIZE_ALL (i.e., no sanitization).
        See RDKit SanitizeFlags documentation for available options.
    aromaticity_model : Optional[AromaticityModel], default=AROMATICITY_MDL
        The aromaticity model to use for determining bond orders and conjugation
        
        Chosen as AROMATICITY_MDL to avoid valence errors which RDKit's default AROMATICITY_RDKIT model
        is known to introduce on certain classes of molecules (e.g. PMDA or indoles such as tryptophan)
        
    Returns
    -------
    cleanmol : Mol
        A sanitized copy of the input molecule
    '''
    # enforce correct typing from deliberate (or accidental) NoneType pass
    cleanmol = Mol(mol)
    if sanitize_ops is None: 
        sanitize_ops = SANITIZE_NONE
    
    # hydrogen handling?
    cleanmol.UpdatePropertyCache(strict=True) # DEV: True is default; decide if this is worth changing down the line
    if add_Hs:
        cleanmol = AddHs(cleanmol)
    
    # aromaticity determination
    if aromaticity_model is not None:
        sanitize_ops = sanitize_ops & ~SanitizeFlags.SANITIZE_KEKULIZE & ~SanitizeFlags.SANITIZE_SETAROMATICITY # prevents final SanitizeMol call from undoing aromatcity model
        Kekulize(cleanmol, clearAromaticFlags=True)
        SetAromaticity(cleanmol, model=aromaticity_model)
           
    # miscellaneous sanitization operations
    SanitizeMol(cleanmol, sanitizeOps=sanitize_ops) # regardless of settings, sanitization should be done last to give greatest likelihodd of molecule validity
    
    return cleanmol