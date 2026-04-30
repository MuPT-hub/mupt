"""Strategy implementations for MuPT -> RDKit export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive
from ...roles import PrimitiveRole
from .._shared.topology import (
    build_saamr_role_topology_index,
    _pdb_resname,
    _resolve_to_atom,
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
    atom_hierarchy_paths: list[list[dict[str, object]]] = field(default_factory=list)
    atom_resids: list[int] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_refs: list[tuple[Primitive, tuple[ConnectorReference, ConnectorReference]]] = field(default_factory=list)
    linker_refs: list[tuple[int, Primitive, ConnectorReference]] = field(default_factory=list)


class RDKitExportStrategy(ABC):
    """Abstract strategy for collecting RDKit-exportable topology data."""

    @abstractmethod
    def validate(self, root: Primitive) -> None:
        """Validate role assignment and hierarchy preconditions for export."""

    @abstractmethod
    def collect_mols(self, root: Primitive, resname_map: dict[str, str]) -> list[RDKitMolData]:
        """Collect one topology dataset per RDKit Mol to build."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name for this strategy."""


class AllAtomRDKitExportStrategy(RDKitExportStrategy):
    """Role-aware all-atom RDKit export strategy."""

    def __init__(self, default_atom_position: Optional[np.ndarray] = None) -> None:
        """Initialize the all-atom RDKit export strategy.

        Parameters
        ----------
        default_atom_position : numpy.ndarray, optional
            Position used for PARTICLE Primitives without shape information.

        Raises
        ------
        ValueError
            If ``default_atom_position`` is not a 3-vector.
        """
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
        """Validate role assignments needed for all-atom RDKit export.

        Parameters
        ----------
        root : Primitive
            Root Primitive expected to use the SAAMR role protocol.

        Raises
        ------
        ValueError
            If the hierarchy is not role-valid for all-atom RDKit export.
        """
        # The index builder performs the role and enclosure validation in one pass.
        build_saamr_role_topology_index(root)

    def collect_mols(self, root: Primitive, resname_map: dict[str, str]) -> list[RDKitMolData]:
        """Collect one RDKit topology dataset per SEGMENT-role node.

        Parameters
        ----------
        root : Primitive
            UNIVERSE-role root of a SAAMR-like hierarchy.
        resname_map : dict[str, str]
            Mapping from Primitive residue labels to 3-character PDB residue names.

        Returns
        -------
        list[RDKitMolData]
            Ordered RDKit topology data, one entry per SEGMENT-role Primitive.
        """
        # Build one DFS-derived index and reuse it for all segment/residue/particle walks.
        index = build_saamr_role_topology_index(root)

        mols_data: list[RDKitMolData] = []
        # Connector chains can be shared across lookups; cache resolved particle endpoints.
        endpoint_cache: dict[tuple[int, object, object], Primitive] = {}

        for segment in index.segments:
            data = RDKitMolData(segment=segment)
            atom_id_to_local: dict[int, int] = {}
            resid_counter = 1

            for residue in index.residues_by_segment[id(segment)]:
                resname = _pdb_resname(residue.label, resname_map)
                for atom in index.particles_by_residue[id(residue)]:
                    atom_id_to_local[id(atom)] = len(data.atoms)
                    data.atoms.append(atom)
                    if atom.shape is not None:
                        data.atom_positions.append(atom.shape.centroid)
                    else:
                        data.atom_positions.append(self.default_atom_position)
                    data.atom_resnames.append(resname)
                    data.atom_insertion_codes.append(str(residue.metadata.get("pdb_insertion_code", "")))
                    data.atom_residue_labels.append(str(residue.label))
                    data.atom_particle_labels.append(str(atom.label))
                    data.atom_hierarchy_paths.append(_serialized_path_from_segment(segment, atom))
                    data.atom_resids.append(resid_counter)
                resid_counter += 1

            bonds_set: set[tuple[int, int]] = set()
            for node in index.bond_nodes:
                # Each output Mol receives only bonds owned by its enclosing SEGMENT.
                if index.segment_of_node[id(node)] is not segment:
                    continue

                for conn_ref_pair in node.internal_connections:
                    conn_ref1, conn_ref2 = sorted(
                        conn_ref_pair,
                        key=lambda cr: (cr.primitive_handle, cr.connector_handle),
                    )
                    atom1 = self._resolve_to_atom_cached(node, conn_ref1, endpoint_cache)
                    atom2 = self._resolve_to_atom_cached(node, conn_ref2, endpoint_cache)
                    idx1 = atom_id_to_local[id(atom1)]
                    idx2 = atom_id_to_local[id(atom2)]
                    bond_pair = tuple(sorted((idx1, idx2)))
                    if bond_pair in bonds_set:
                        # Multiple hierarchy-level paths can resolve to the same atom pair.
                        continue

                    data.bonds.append(bond_pair)
                    data.bond_refs.append((node, (conn_ref1, conn_ref2)))
                    bonds_set.add(bond_pair)

            for conn_ref in segment.external_connectors.values():
                atom = self._resolve_to_atom_cached(segment, conn_ref, endpoint_cache)
                data.linker_refs.append((atom_id_to_local[id(atom)], segment, conn_ref))

            if data.bonds:
                sorted_bonds = sorted(
                    zip(data.bonds, data.bond_refs), key=lambda pair: pair[0]
                )
                data.bonds = [bond for bond, _ in sorted_bonds]
                data.bond_refs = [bond_ref for _, bond_ref in sorted_bonds]

            mols_data.append(data)

        return mols_data

    @staticmethod
    def _resolve_to_atom_cached(
        parent: Primitive,
        conn_ref: ConnectorReference,
        cache: dict[tuple[int, object, object], Primitive],
    ) -> Primitive:
        """Resolve a connector reference to an atom using a per-export cache."""
        cache_key = (id(parent), conn_ref.primitive_handle, conn_ref.connector_handle)
        if cache_key not in cache:
            # Resolve through arbitrary intermediate hierarchy only once per connector ref.
            cache[cache_key] = _resolve_to_atom(parent, conn_ref)
        return cache[cache_key]


def _child_handle(parent: Primitive, child: Primitive) -> tuple[object, int]:
    """Return the parent-local handle for a known child Primitive."""
    for handle, candidate in parent.children_by_handle.items():
        if candidate is child:
            return handle
    raise ValueError(f"Child '{child.label}' is not attached to parent '{parent.label}'")


def _serialized_path_from_segment(segment: Primitive, atom: Primitive) -> list[dict[str, object]]:
    """Return a stable SEGMENT-to-PARTICLE hierarchy path for one atom."""
    segment_idxs = [idx for idx, node in enumerate(atom.path) if node is segment]
    if not segment_idxs:
        raise ValueError(f"Atom '{atom.label}' is not enclosed by segment '{segment.label}'")
    segment_idx = segment_idxs[0]

    path = []
    for node in atom.path[segment_idx:]:
        if node is segment:
            handle_label, handle_index = str(node.label), 0
        else:
            handle_label, handle_index = _child_handle(node.parent, node)
        path.append(
            {
                "role": node.role.value,
                "label": str(node.label),
                "handle_label": str(handle_label),
                "handle_index": int(handle_index),
                "is_atom": bool(node.is_atom),
            }
        )

    if path[-1]["role"] != PrimitiveRole.PARTICLE.value:
        raise ValueError("Serialized RDKit atom path must terminate at a PARTICLE role")
    return path
