'''Interfaces for SMILES, SMARTS, BIGSMILES, and other line notations'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Hashable, Optional

from rdkit.Chem.rdmolops import (
    SanitizeFlags,
    SANITIZE_ALL,
    AromaticityModel,
    AROMATICITY_MDL,
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
from ..chemistry.sanitization import sanitized_mol


def primitive_from_smiles(
    smiles : str, 
    label : Optional[Hashable]=None,
    embed_positions : bool=False,
    ensure_explicit_Hs : bool=True,
    aromaticity_model : AromaticityModel=AROMATICITY_MDL,
    sanitize_ops : SanitizeFlags=SANITIZE_ALL,
    smiles_reader_params=DEFAULT_SMILES_READ_PARAMS,
    smiles_writer_params=DEFAULT_SMILES_WRITE_PARAMS, 
) -> Primitive:
    '''Create a Primitive from a SMILES string, optionally embedding positions if selected'''
    rdmol = sanitized_mol(
        MolFromSmiles(smiles, params=smiles_reader_params),
        add_Hs=ensure_explicit_Hs,
        sanitize_ops=sanitize_ops,
        aromaticity_model=aromaticity_model,
    )
    conformer_idx : Optional[int] = None
    if embed_positions:
        conformer_idx = EmbedMolecule(rdmol, clearConfs=False) # NOTE: don't clobber existing conformers for safety (though new Mol shouldn't have any anyway)
    
    return primitive_from_rdkit(
        rdmol,
        conformer_idx=conformer_idx,
        label=label,
        smiles_writer_params=smiles_writer_params, # DEV: needed to generate SMILES from mol in case no explicit label is provided
    )

def primitive_to_smiles(
    primitive : Primitive,
    smiles_write_params : Optional[SmilesWriteParams]=DEFAULT_SMILES_WRITE_PARAMS,
) -> str:
    '''Convert a Primitive to a SMILES string'''
    return MolToSmiles(
        primitive_to_rdkit(primitive),
        params=smiles_write_params,
    )