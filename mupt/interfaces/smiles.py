"""Interfaces for SMILES, SMARTS, BIGSMILES, and other line notations"""

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

from .rdkit import primitive_from_rdkit, rdkit_mol_from_primitive
from ..mupr.primitives import Primitive, SupportsChildren
from ..chemistry.smiles import DEFAULT_SMILES_READ_PARAMS, DEFAULT_SMILES_WRITE_PARAMS
from ..chemistry.rdkit.sanitization import sanitized_mol
from ..chemistry.rdkit.linkers import embed_with_linker_placeholders


def primitive_from_smiles(
    smiles: str,
    label: Optional[Hashable] = None,
    embed_positions: bool = True,
    embed_random_seed: int = -1,
    ensure_explicit_Hs: bool = True,
    aromaticity_model: AromaticityModel = AROMATICITY_MDL,
    sanitize_ops: SanitizeFlags = SANITIZE_ALL,
    smiles_reader_params=DEFAULT_SMILES_READ_PARAMS,
    smiles_writer_params=DEFAULT_SMILES_WRITE_PARAMS,
    **kwargs,
) -> SupportsChildren:
    """
    Create a Primitive from a SMILES string

    By default, embeds 3D coordinates for all atoms (and Connectors) using RDKit's
    ETKDG embedding (see embed_with_linker_placeholders); pass embed_random_seed >= 0
    for reproducible coordinates
    """
    rdmol = sanitized_mol(
        MolFromSmiles(smiles, params=smiles_reader_params),
        add_Hs=ensure_explicit_Hs,
        sanitize_ops=sanitize_ops,
        aromaticity_model=aromaticity_model,
    )
    conformer_idx: Optional[int] = None
    if embed_positions:
        conformer_idx = embed_with_linker_placeholders(rdmol, random_seed=embed_random_seed)
        if conformer_idx < 0:
            raise ValueError(f"RDKit failed to embed coordinates for SMILES {smiles!r}")

    return primitive_from_rdkit(
        rdmol,
        conformer_idx=conformer_idx,
        label=label,
        # DEV: needed to generate SMILES from mol in case no explicit label is provided
        smiles_writer_params=smiles_writer_params,
        **kwargs,
    )


def primitive_to_smiles(
    primitive: Primitive,
    smiles_write_params: Optional[SmilesWriteParams] = DEFAULT_SMILES_WRITE_PARAMS,
) -> str:
    """Convert a Primitive to a SMILES string"""
    return MolToSmiles(
        rdkit_mol_from_primitive(primitive),
        params=smiles_write_params,
    )
