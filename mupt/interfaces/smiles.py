'''Interfaces for SMILES, SMARTS, BIGSMILES, and other line notations'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Hashable, Optional

from rdkit.Chem.rdmolops import (
    AddHs,
    SanitizeMol,
    SanitizeFlags,
    SANITIZE_ALL,
)
from rdkit.Chem.rdmolfiles import (
    MolFromSmiles,
    MolToSmiles,
    SmilesWriteParams,
)
from rdkit.Chem.rdDistGeom import EmbedMolecule

from .rdkit import primitive_from_rdkit, primitive_to_rdkit
from ..mupr.primitives import Primitive
from ..chemistry.smiles import DEFAULT_SMILES_READ_PARAMS, DEFAULT_SMILES_WRITE_PARAMS


def primitive_from_smiles(
    smiles : str, 
    ensure_explicit_Hs : bool=True,
    embed_positions : bool=False,
    sanitize_ops : SanitizeFlags=SANITIZE_ALL,
    label : Optional[Hashable]=None,
    smiles_writer_params=DEFAULT_SMILES_WRITE_PARAMS,
) -> Primitive:
    '''Create a Primitive from a SMILES string, optionally embedding positions if selected'''
    rdmol = MolFromSmiles(smiles, sanitize=False)
    if ensure_explicit_Hs:
        rdmol.UpdatePropertyCache() # allow Hs to be added without sanitizating twice
        rdmol = AddHs(rdmol)
    SanitizeMol(rdmol, sanitizeOps=sanitize_ops)
    
    conformer_idx : Optional[int] = None
    if embed_positions:
        conformer_idx = EmbedMolecule(rdmol, clearConfs=False) # NOTE: don't clobber existing conformers for safety (though new Mol shouldn't have any anyway)
    
    return primitive_from_rdkit(
        rdmol,
        conformer_idx=conformer_idx,
        label=label,
        smiles_writer_params=smiles_writer_params,
    )

def primitive_to_smiles(
    primitive : Primitive,
    smiles_params : Optional[SmilesWriteParams]=DEFAULT_SMILES_READ_PARAMS,
) -> str:
    '''Convert a Primitive to a SMILES string'''
    return MolToSmiles(
        primitive_to_rdkit(primitive),
        params=smiles_params,
    )