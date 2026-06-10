"""Temporary SDF export bridge for MuPT interoperability workflows.

This helper writes role-aware MuPT hierarchies to RDKit-readable multi-record
SDF files for downstream setup and analysis tools such as OpenFF. It is a
temporary export-only bridge, not canonical MuPT persistence and not a general
RDKit file-format wrapper. Keep it easy to remove before a stable native MuPT
persistence format exists.

The writer preserves per-segment records and MuPT/RDKit atom metadata by
preparing RDKit SDF atom-property lists before writing. It does not deserialize
SDF, nor attempt to losslessly recover the originating MuPT hierarchy.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from pathlib import Path
from typing import Optional

import numpy as np

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import CreateAtomStringPropertyList, SDWriter

from ..geometry.arraytypes import Shape
from ..interfaces.rdkit.exporters import MUPT_RDKIT_ATOM_PROPS, primitive_to_rdkit_mols
from ..interfaces.rdkit.strategies import RDKitExportStrategy
from ..mutils.filepaths.pathutils import asstrpath
from ..mupr.primitives import Primitive


MUPT_SDF_ATOM_PROPS = MUPT_RDKIT_ATOM_PROPS


def prepare_mupt_sdf_atom_props(mol: Mol) -> None:
    """Store MuPT atom props as SDF atom-property lists before writing."""
    for prop_name in MUPT_SDF_ATOM_PROPS:
        CreateAtomStringPropertyList(mol, prop_name, missingValueMarker="NA", lineSize=10000)


def write_primitive_to_sdf(
    primitive: Primitive,
    path: str | Path,
    resname_map: dict[str, str],
    default_atom_position: Optional[np.ndarray[Shape[3], float]] = None,
    strategy: Optional[RDKitExportStrategy] = None,
) -> int:
    """Stream a role-annotated Primitive hierarchy to a multi-record SDF file.

    Returns the number of SDF records written. The implementation consumes one
    generated RDKit Mol at a time so large systems do not retain every segment
    molecule in memory.
    """
    records = 0
    writer = SDWriter(asstrpath(path))
    try:
        for mol in primitive_to_rdkit_mols(
            primitive,
            resname_map=resname_map,
            default_atom_position=default_atom_position,
            strategy=strategy,
        ):
            prepare_mupt_sdf_atom_props(mol)
            writer.write(mol)
            records += 1
    finally:
        writer.close()
    return records


write_primitive_to_mupt_sdf = write_primitive_to_sdf
