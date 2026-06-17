"""Temporary SDF export bridge for MuPT interoperability workflows.

This helper writes role-aware MuPT hierarchies to RDKit-readable multi-record
SDF files for downstream setup and analysis tools such as OpenFF. It is a
temporary bridge, not canonical MuPT persistence and not a general RDKit
file-format wrapper. Keep it easy to remove before a stable native MuPT
persistence format exists.

The writer preserves per-segment records and MuPT/RDKit atom metadata by
preparing RDKit SDF atom-property lists before writing. The reader rebuilds
one SEGMENT hierarchy per SDF record; covalent bonds between records are not
representable in this temporary per-segment format. SDF metadata is record-level,
so imported record metadata is stored on rebuilt SEGMENT nodes.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from collections.abc import Iterator
import os
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    CreateAtomStringPropertyList,
    SDMolSupplier,
    SDWriter,
)

from ..chemistry.conversion import rdkit_atom_to_element
from ..chemistry.sanitization import sanitized_mol
from ..geometry.arraytypes import Shape
from ..geometry.shapes import PointCloud
from ..interfaces.rdkit.exporters import MUPT_RDKIT_ATOM_PROPS, primitive_to_rdkit_mols
from ..interfaces.rdkit.strategies import RDKitExportStrategy
from ..mutils.filepaths.pathutils import asstrpath
from ..mupr.connection import AttachmentPoint, Connector
from ..mupr.primitives import Primitive
from ..roles import PrimitiveRole


MUPT_SDF_ATOM_PROPS = MUPT_RDKIT_ATOM_PROPS
MUPT_SDF_SUFFIX = ".mupt.sdf"
MUPT_SDF_ATOM_PROP_PREFIX = "atom.prop."


def prepare_mupt_sdf_atom_props(mol: Mol) -> None:
    """Store MuPT atom props as SDF atom-property lists before writing."""
    for prop_name in MUPT_SDF_ATOM_PROPS:
        CreateAtomStringPropertyList(
            mol,
            prop_name,
            missingValueMarker="NA",
            lineSize=10000,
        )


def _mupt_sdf_path(path: str | Path) -> Path:
    """Return ``path`` normalized to the temporary MuPT SDF suffix."""
    path = Path(path)
    path_str = asstrpath(path)
    if path_str.endswith(MUPT_SDF_SUFFIX):
        return path
    if path.suffix == ".sdf":
        return path.with_suffix(MUPT_SDF_SUFFIX)
    return Path(f"{path_str}{MUPT_SDF_SUFFIX}")


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
    target_path = _mupt_sdf_path(path)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=asstrpath(target_path.parent),
    )
    os.close(temp_fd)

    records = 0
    completed = False
    temp_path = Path(temp_name)
    try:
        writer = SDWriter(asstrpath(temp_path))
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
        temp_path.replace(target_path)
        completed = True
    finally:
        if not completed and temp_path.exists():
            temp_path.unlink()
    return records


def _required_atom_prop(atom, prop_name: str) -> str:
    """Fetch a required MuPT atom property from an RDKit atom."""
    if not atom.HasProp(prop_name):
        raise ValueError(
            f"MuPT SDF atom {atom.GetIdx()} lacks required property '{prop_name}'"
        )
    return atom.GetProp(prop_name)


def _atom_position(
    mol: Mol,
    atom_idx: int,
    conformer_idx: Optional[int],
) -> Optional[np.ndarray]:
    """Return one atom position from an RDKit conformer, if present."""
    if conformer_idx is None:
        return None
    return np.array(mol.GetConformer(conformer_idx).GetAtomPosition(atom_idx), dtype=float)


def _particle_from_sdf_atom(
    mol: Mol,
    atom_idx: int,
    conformer_idx: Optional[int],
) -> Primitive:
    """Build a PARTICLE primitive from one MuPT SDF atom."""
    atom = mol.GetAtomWithIdx(atom_idx)
    particle = Primitive(
        element=rdkit_atom_to_element(atom),
        label=_required_atom_prop(atom, "mupt_particle_label"),
        metadata=atom.GetPropsAsDict(includePrivate=True, includeComputed=False),
        role=PrimitiveRole.PARTICLE,
    )
    if (position := _atom_position(mol, atom_idx, conformer_idx)) is not None:
        particle.shape = PointCloud(positions=position)
    return particle


def _attachment_from_sdf_atom(atom) -> AttachmentPoint:
    """Return the lightweight attachment identity used for MuPT SDF bonds.

    The temporary importer only needs connector data sufficient to re-export the
    same per-record SDF topology. Matching the RDKit component helper's atom
    index/symbol identity keeps round-trip connector labels stable without
    paying to rebuild SMARTS queries.
    """
    atom_idx = atom.GetIdx()
    atom_symbol = atom.GetSymbol()
    return AttachmentPoint(
        attachables={atom_idx, atom_symbol},
        attachment=atom_idx,
    )


def _sdf_bond_metadata(bond) -> dict:
    """Return bond metadata preserved by temporary MuPT SDF import."""
    return {
        "bond_stereo": bond.GetStereo(),
        "bond_stereo_atoms": tuple(bond.GetStereoAtoms()),
        **bond.GetPropsAsDict(includePrivate=True, includeComputed=False),
    }


def _connector_from_sdf_bond(
    mol: Mol,
    from_atom_idx: int,
    to_atom_idx: int,
    conformer_idx: Optional[int],
) -> Connector:
    """Build one directed connector from an SDF bond for round-trip export.

    This is deliberately narrower than ``connector_between_rdatoms``. The
    temporary MuPT SDF reader only rebuilds enough connector state to emit the
    same segment records again, so it preserves bond type, bond metadata, and
    anchor/linker coordinates while skipping SMARTS generation and tangent-vector
    reconstruction. Those skipped fields are expensive and are not consumed by
    the temporary SDF round-trip exporter.
    """
    bond = mol.GetBondBetweenAtoms(from_atom_idx, to_atom_idx)
    if bond is None:
        raise ValueError(
            f"MuPT SDF atoms {from_atom_idx} and {to_atom_idx} are not bonded"
        )

    connector = Connector(
        anchor=_attachment_from_sdf_atom(mol.GetAtomWithIdx(from_atom_idx)),
        linker=_attachment_from_sdf_atom(mol.GetAtomWithIdx(to_atom_idx)),
        bondtype=bond.GetBondType(),
        metadata=_sdf_bond_metadata(bond),
    )
    if conformer_idx is not None:
        connector.anchor.position = _atom_position(mol, from_atom_idx, conformer_idx)
        connector.linker.position = _atom_position(mol, to_atom_idx, conformer_idx)
    return connector


def _has_particle_props(atom) -> bool:
    """Return whether an SDF atom carries MuPT particle identity props."""
    return atom.HasProp("mupt_particle_index") and atom.HasProp("mupt_particle_label")


def _is_external_linker_atom(atom) -> bool:
    """Return whether an atom is an exported MuPT external connector linker."""
    return atom.GetAtomicNum() == 0 and not _has_particle_props(atom)


def _record_metadata(mol: Mol) -> dict:
    """Return non-atom-list SDF record metadata for segment-level preservation."""
    return {
        key: value
        for key, value in mol.GetPropsAsDict(includePrivate=True, includeComputed=False).items()
        if not key.startswith(MUPT_SDF_ATOM_PROP_PREFIX)
    }


def _build_segment_from_mol(mol: Mol) -> Primitive:
    """Rebuild one SEGMENT hierarchy from one MuPT SDF record."""
    conformer_idx = 0 if mol.GetNumConformers() else None
    for atom in mol.GetAtoms():
        if not (_has_particle_props(atom) or _is_external_linker_atom(atom)):
            raise ValueError(
                f"MuPT SDF atom {atom.GetIdx()} lacks required MuPT particle props"
            )
    particle_atoms = [atom for atom in mol.GetAtoms() if _has_particle_props(atom)]
    if not particle_atoms:
        raise ValueError("MuPT SDF record contains no PARTICLE atoms")

    segment_label = _required_atom_prop(particle_atoms[0], "mupt_segment_label")
    segment_index = _required_atom_prop(particle_atoms[0], "mupt_segment_index")
    for atom in particle_atoms[1:]:
        if (
            _required_atom_prop(atom, "mupt_segment_label") != segment_label
            or _required_atom_prop(atom, "mupt_segment_index") != segment_index
        ):
            raise ValueError("MuPT SDF record has inconsistent SEGMENT identity")
    segment = Primitive(
        label=segment_label,
        metadata=_record_metadata(mol),
        role=PrimitiveRole.SEGMENT,
    )

    residue_data = {}
    atom_to_residue_index = {}
    for atom in particle_atoms:
        atom_idx = atom.GetIdx()
        residue_index = int(_required_atom_prop(atom, "mupt_residue_index"))
        atom_to_residue_index[atom_idx] = residue_index
        residue_label = _required_atom_prop(atom, "mupt_residue_label")
        if residue_index in residue_data:
            if residue_data[residue_index]["label"] != residue_label:
                raise ValueError(
                    "MuPT SDF record has conflicting RESIDUE labels for "
                    f"index {residue_index}"
                )
        else:
            residue_data[residue_index] = {"label": residue_label, "atoms": []}
        residue_data[residue_index]["atoms"].append(atom_idx)

    residue_primitives = {}
    atom_handles = {}
    atom_connector_handles = {}
    for residue_index in sorted(residue_data):
        data = residue_data[residue_index]
        residue = Primitive(
            label=data["label"],
            metadata={"mupt_residue_index": residue_index},
            role=PrimitiveRole.RESIDUE,
        )
        for atom_idx in sorted(
            data["atoms"],
            key=lambda idx: int(
                _required_atom_prop(mol.GetAtomWithIdx(idx), "mupt_particle_index")
            ),
        ):
            particle = _particle_from_sdf_atom(mol, atom_idx, conformer_idx)
            for nb_atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                conn = _connector_from_sdf_bond(
                    mol,
                    atom_idx,
                    nb_atom.GetIdx(),
                    conformer_idx=conformer_idx,
                )
                atom_connector_handles[(atom_idx, nb_atom.GetIdx())] = (
                    particle.register_connector(conn)
                )
            atom_handles[atom_idx] = residue.attach_child(particle)
        residue_primitives[residue_index] = residue

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx not in atom_to_residue_index or end_idx not in atom_to_residue_index:
            if not (
                _is_external_linker_atom(mol.GetAtomWithIdx(begin_idx))
                or _is_external_linker_atom(mol.GetAtomWithIdx(end_idx))
            ):
                raise ValueError("MuPT SDF bond references an atom without particle props")
            continue
        begin_residue_index = atom_to_residue_index[begin_idx]
        end_residue_index = atom_to_residue_index[end_idx]
        if begin_residue_index == end_residue_index:
            residue_primitives[begin_residue_index].connect_children(
                atom_handles[begin_idx],
                atom_connector_handles[(begin_idx, end_idx)],
                atom_handles[end_idx],
                atom_connector_handles[(end_idx, begin_idx)],
            )

    residue_handles = {}
    for residue_index in sorted(residue_primitives):
        residue_handles[residue_index] = segment.attach_child(residue_primitives[residue_index])

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx not in atom_to_residue_index or end_idx not in atom_to_residue_index:
            if not (
                _is_external_linker_atom(mol.GetAtomWithIdx(begin_idx))
                or _is_external_linker_atom(mol.GetAtomWithIdx(end_idx))
            ):
                raise ValueError("MuPT SDF bond references an atom without particle props")
            continue
        begin_residue_index = atom_to_residue_index[begin_idx]
        end_residue_index = atom_to_residue_index[end_idx]
        if begin_residue_index == end_residue_index:
            continue
        begin_atom = mol.GetAtomWithIdx(begin_idx)
        end_atom = mol.GetAtomWithIdx(end_idx)
        if _required_atom_prop(begin_atom, "mupt_segment_index") != _required_atom_prop(
            end_atom,
            "mupt_segment_index",
        ):
            raise ValueError("MuPT SDF bond crosses SEGMENT records")
        begin_residue = residue_primitives[begin_residue_index]
        end_residue = residue_primitives[end_residue_index]
        begin_residue_conn = begin_residue.external_connectors_on_child(atom_handles[begin_idx])[
            atom_connector_handles[(begin_idx, end_idx)]
        ]
        end_residue_conn = end_residue.external_connectors_on_child(atom_handles[end_idx])[
            atom_connector_handles[(end_idx, begin_idx)]
        ]
        segment.connect_children(
            residue_handles[begin_residue_index],
            begin_residue_conn,
            residue_handles[end_residue_index],
            end_residue_conn,
        )

    return segment


def iter_primitives_from_mupt_sdf(
    path: str | Path,
    sanitize: bool = False,
) -> Iterator[Primitive]:
    """Yield one rebuilt SEGMENT primitive per temporary MuPT SDF record.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.mupt.sdf`` file written by :func:`write_primitive_to_sdf`.
    sanitize : bool, default=False
        Whether MuPT should sanitize records before rebuilding them. Records are
        always read from RDKit unsanitized first; when requested, sanitization is
        applied with :func:`mupt.chemistry.sanitization.sanitized_mol`, including
        MuPT's default MDL aromaticity convention.
        Coordinates are imported directly from RDKit conformers using the same
        distance convention as the source SDF records, conventionally angstroms.

    Yields
    -------
    Primitive
        Rebuilt ``SEGMENT -> RESIDUE -> PARTICLE`` hierarchy for one SDF record.
        Per-record SDF metadata is preserved on rebuilt SEGMENT nodes. Bonds
        between records are unsupported because this temporary format writes one
        segment per SDF record and has no cross-record bond representation.

    Raises
    ------
    ValueError
        If RDKit cannot parse a record or if MuPT SDF identity metadata is
        missing or inconsistent.
    """
    path = _mupt_sdf_path(path)
    supplier = SDMolSupplier(
        asstrpath(path),
        removeHs=False,
        sanitize=False,
    )
    for record_idx, mol in enumerate(supplier):
        if mol is None:
            raise ValueError(f"Could not parse MuPT SDF record {record_idx} from '{path}'")
        if sanitize:
            mol = sanitized_mol(mol)
        yield _build_segment_from_mol(mol)


def primitive_from_mupt_sdf(path: str | Path, sanitize: bool = False) -> Primitive:
    """Read a temporary MuPT SDF file into a role-aware Primitive hierarchy.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.mupt.sdf`` file written by :func:`write_primitive_to_sdf`.
    sanitize : bool, default=False
        Whether MuPT should sanitize records before rebuilding them. Records are
        always read from RDKit unsanitized first; when requested, sanitization is
        applied with :func:`mupt.chemistry.sanitization.sanitized_mol`, including
        MuPT's default MDL aromaticity convention.
        Coordinates are imported directly from RDKit conformers using the same
        distance convention as the source SDF records, conventionally angstroms.

    Returns
    -------
    Primitive
        Rebuilt ``UNIVERSE -> SEGMENT -> RESIDUE -> PARTICLE`` hierarchy.
        Per-record SDF metadata is preserved on rebuilt SEGMENT nodes because
        SDF has no file-level metadata scope for reconstructing root metadata.
    """
    universe = Primitive(label="MuPT SDF", role=PrimitiveRole.UNIVERSE)
    for segment in iter_primitives_from_mupt_sdf(path, sanitize=sanitize):
        universe.attach_child(segment)
    return universe


write_primitive_to_mupt_sdf = write_primitive_to_sdf
