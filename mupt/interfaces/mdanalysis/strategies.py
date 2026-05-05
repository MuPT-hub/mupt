"""Strategy implementations for MuPT -> MDAnalysis export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...mupr.primitives import Primitive
from .._shared.topology import (
    _bond_order_from_conn_ref,
    _pdb_resname,
    build_saamr_role_topology_index,
    connector_reference_sort_key,
    resolve_to_atom_cached,
)


@dataclass
class MDATopologyData:
    """Container for topology arrays/lists used to build an MDAnalysis Universe."""

    atom_elements: list[str] = field(default_factory=list)
    atom_names: list[str] = field(default_factory=list)
    atom_positions: list[list[float]] = field(default_factory=list)
    atom_resindex: list[int] = field(default_factory=list)
    atom_segindex: list[int] = field(default_factory=list)
    residue_names: list[str] = field(default_factory=list)
    residue_segindex: list[int] = field(default_factory=list)
    residue_ids: list[int] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_orders: list[float] = field(default_factory=list)
    bonds_set: set[tuple[int, int]] = field(default_factory=set)
    num_segments: int = 0


class MDAExportStrategy(ABC):
    """Abstract strategy for collecting MDAnalysis-exportable topology data."""

    @abstractmethod
    def validate(self, root: Primitive) -> None:
        """Validate role assignment and hierarchy preconditions for export."""

    @abstractmethod
    def collect_topology(self, root: Primitive, resname_map: dict[str, str]) -> MDATopologyData:
        """Collect topology attributes from a Primitive hierarchy."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name for this strategy."""


class AllAtomExportStrategy(MDAExportStrategy):
    """All-atom export strategy based on role-aware hierarchy traversal.

    Although only the four SAAMR roles are recognized (UNIVERSE,
    SEGMENT, RESIDUE, PARTICLE), this strategy supports trees of arbitrary
    depth. Intermediate nodes between role-tagged levels (e.g., a "domain"
    grouping between UNIVERSE and SEGMENT) are traversed transparently by the
    shared SAAMR role index and carry ``PrimitiveRole.UNASSIGNED``.
    """

    def __init__(
        self,
        default_atom_position: Optional[np.ndarray] = None,
    ) -> None:
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
        """Validate role assignments needed for all-atom export."""
        build_saamr_role_topology_index(root)

    def collect_topology(self, root: Primitive, resname_map: dict[str, str]) -> MDATopologyData:
        """Walk the hierarchy once and gather MDAnalysis topology arrays/lists."""
        index = build_saamr_role_topology_index(root)
        data = MDATopologyData()

        particles = [
            atom
            for segment in index.segments
            for residue in index.residues_by_segment[id(segment)]
            for atom in index.particles_by_residue[id(residue)]
        ]
        n_atoms = len(particles)
        atom_id_to_global: dict[int, int] = {
            id(atom): idx for idx, atom in enumerate(particles)
        }

        data.atom_resindex = [0] * n_atoms
        data.atom_segindex = [0] * n_atoms

        for atom in particles:
            atom_symbol = atom.element.symbol
            data.atom_elements.append(atom_symbol)
            data.atom_names.append(atom_symbol)

            if atom.shape is not None:
                data.atom_positions.append(list(atom.shape.centroid))
            else:
                data.atom_positions.append(list(self.default_atom_position))

        data.num_segments = len(index.segments)
        for seg_idx, segment in enumerate(index.segments):
            for resid_counter, residue in enumerate(
                index.residues_by_segment[id(segment)],
                start=1,
            ):
                data.residue_names.append(_pdb_resname(residue.label, resname_map))
                data.residue_segindex.append(seg_idx)
                data.residue_ids.append(resid_counter)

                res_global_idx = len(data.residue_names) - 1
                for atom in index.particles_by_residue[id(residue)]:
                    atom_global_idx = atom_id_to_global[id(atom)]
                    data.atom_resindex[atom_global_idx] = res_global_idx
                    data.atom_segindex[atom_global_idx] = seg_idx

        endpoint_cache: dict[tuple[int, object, object], Primitive] = {}
        for node in index.bond_nodes:
            for conn_ref_pair in node.internal_connections:
                ref_list = sorted(
                    conn_ref_pair,
                    key=connector_reference_sort_key,
                )
                conn_ref1, conn_ref2 = ref_list
                atom1 = resolve_to_atom_cached(node, conn_ref1, endpoint_cache)
                atom2 = resolve_to_atom_cached(node, conn_ref2, endpoint_cache)

                idx1 = atom_id_to_global[id(atom1)]
                idx2 = atom_id_to_global[id(atom2)]
                bond_pair = tuple(sorted((idx1, idx2)))
                if bond_pair in data.bonds_set:
                    # Duplicate atom-pair: MuPT's bondable_with() enforces
                    # symmetric bondtype, so the same pair always resolves
                    # to the same bond order — safe to skip.
                    continue

                data.bonds.append(bond_pair)
                data.bonds_set.add(bond_pair)
                data.bond_orders.append(_bond_order_from_conn_ref(node, conn_ref1))

        # Sort bonds for deterministic output (internal_connections is a set,
        # so iteration order is nondeterministic without explicit sorting)
        if data.bonds:
            sorted_pairs = sorted(
                zip(data.bonds, data.bond_orders), key=lambda pair: pair[0]
            )
            data.bonds = [b for b, _ in sorted_pairs]
            data.bond_orders = [o for _, o in sorted_pairs]

        return data


class CoarseGrainedExportStrategy(MDAExportStrategy):
    """Placeholder strategy for future coarse-grained MDAnalysis export."""

    @property
    def label(self) -> str:
        """Human-readable strategy name."""
        raise NotImplementedError("Coarse-grained export is not yet implemented")

    def validate(self, root: Primitive) -> None:
        """Validate hierarchy for coarse-grained export."""
        raise NotImplementedError("Coarse-grained export is not yet implemented")

    def collect_topology(self, root: Primitive, resname_map: dict[str, str]) -> MDATopologyData:
        """Collect topology for coarse-grained export."""
        raise NotImplementedError("Coarse-grained export is not yet implemented")
