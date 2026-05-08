"""Strategy implementations for MuPT -> RDKit export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive
from .._shared.topology import (
    _pdb_resname,
    build_saamr_role_topology_index,
    connector_reference_sort_key,
    iter_saamr_residue_records,
    resolve_to_atom_cached,
)


@dataclass
class RDKitMolData:
    """Container for one segment's RDKit-exportable topology data."""

    segment: Primitive
    atoms: list[Primitive] = field(default_factory=list)
    atom_positions: list[np.ndarray] = field(default_factory=list)
    atom_resnames: list[str] = field(default_factory=list)
    atom_insertion_codes: list[str] = field(default_factory=list)
    atom_residue_labels: list[str] = field(default_factory=list)
    atom_particle_labels: list[str] = field(default_factory=list)
    atom_resids: list[int] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_refs: list[
        tuple[Primitive, tuple[ConnectorReference, ConnectorReference]]
    ] = field(default_factory=list)
    linker_refs: list[tuple[int, Primitive, ConnectorReference]] = field(default_factory=list)


class RDKitExportStrategy(ABC):
    """Abstract strategy for collecting RDKit-exportable topology data."""

    @abstractmethod
    def validate(self, root: Primitive) -> None:
        """Validate role assignment and hierarchy preconditions for export."""

    @abstractmethod
    def iter_mol_data(self, root: Primitive, resname_map: dict[str, str]) -> Iterator[RDKitMolData]:
        """Yield one topology dataset per RDKit Mol to build."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name for this strategy."""


class AllAtomRDKitExportStrategy(RDKitExportStrategy):
    """Role-aware all-atom RDKit export strategy."""

    def __init__(self, default_atom_position: Optional[np.ndarray] = None) -> None:
        if default_atom_position is None:
            self.default_atom_position = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            default_atom_position = np.asarray(default_atom_position, dtype=float)
            if default_atom_position.shape != (3,):
                raise ValueError('default_atom_position must be a 3-dimensional vector')
            self.default_atom_position = default_atom_position

    @property
    def label(self) -> str:
        """Human-readable strategy name."""
        return "All-atom"

    def validate(self, root: Primitive) -> None:
        """Validate role assignments needed for all-atom RDKit export."""
        build_saamr_role_topology_index(root)

    def iter_mol_data(self, root: Primitive, resname_map: dict[str, str]) -> Iterator[RDKitMolData]:
        """Yield one RDKit topology dataset per SEGMENT-role node."""
        index = build_saamr_role_topology_index(root)
        endpoint_cache: dict[tuple[int, object, object], Primitive] = {}
        residue_records_by_segment = {id(segment): [] for segment in index.segments}
        for residue_record in iter_saamr_residue_records(index):
            residue_records_by_segment[id(residue_record.segment)].append(residue_record)

        for segment in index.segments:
            data = RDKitMolData(segment=segment)
            atom_id_to_local: dict[int, int] = {}

            for residue_record in residue_records_by_segment[id(segment)]:
                resname = _pdb_resname(residue_record.residue.label, resname_map)
                for atom in residue_record.particles:
                    atom_id_to_local[id(atom)] = len(data.atoms)
                    data.atoms.append(atom)
                    if atom.shape is not None:
                        data.atom_positions.append(atom.shape.centroid)
                    else:
                        data.atom_positions.append(self.default_atom_position)
                    data.atom_resnames.append(resname)
                    data.atom_insertion_codes.append(str(residue_record.residue.metadata.get("pdb_insertion_code", "")))
                    data.atom_residue_labels.append(str(residue_record.residue.label))
                    data.atom_particle_labels.append(str(atom.label))
                    data.atom_resids.append(residue_record.residue_idx)

            bonds_set: set[tuple[int, int]] = set()
            for node in index.bond_nodes_by_segment[id(segment)]:
                for conn_ref_pair in node.internal_connections:
                    conn_ref1, conn_ref2 = sorted(
                        conn_ref_pair,
                        key=connector_reference_sort_key,
                    )
                    atom1 = resolve_to_atom_cached(node, conn_ref1, endpoint_cache)
                    atom2 = resolve_to_atom_cached(node, conn_ref2, endpoint_cache)
                    idx1 = atom_id_to_local[id(atom1)]
                    idx2 = atom_id_to_local[id(atom2)]
                    bond_pair = tuple(sorted((idx1, idx2)))
                    if bond_pair in bonds_set:
                        raise ValueError(
                            "Multiple MuPT internal connections resolve to the same "
                            f"RDKit atom pair {bond_pair} in SEGMENT '{segment.label}'. "
                            "Role-aware export cannot choose which connector metadata "
                            "to preserve."
                        )

                    data.bonds.append(bond_pair)
                    data.bond_refs.append((node, (conn_ref1, conn_ref2)))
                    bonds_set.add(bond_pair)

            for conn_ref in segment.external_connectors.values():
                atom = resolve_to_atom_cached(segment, conn_ref, endpoint_cache)
                data.linker_refs.append((atom_id_to_local[id(atom)], segment, conn_ref))

            if data.bonds:
                sorted_bonds = sorted(
                    zip(data.bonds, data.bond_refs), key=lambda pair: pair[0]
                )
                data.bonds = [bond for bond, _ in sorted_bonds]
                data.bond_refs = [bond_ref for _, bond_ref in sorted_bonds]

            yield data
