"""Strategy implementations for MuPT -> RDKit export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from anytree import PreOrderIter
import numpy as np

from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive
from ...roles import PrimitiveRole
from .._shared.topology import _pdb_resname, _resolve_to_atom


@dataclass
class RDKitMolData:
    """Container for one segment's RDKit-exportable topology data."""

    segment: Primitive
    atoms: list[Primitive] = field(default_factory=list)
    atom_positions: list[np.ndarray] = field(default_factory=list)
    atom_resnames: list[str] = field(default_factory=list)
    atom_residue_labels: list[str] = field(default_factory=list)
    atom_particle_labels: list[str] = field(default_factory=list)
    atom_resids: list[int] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_refs: list[tuple[Primitive, ConnectorReference]] = field(default_factory=list)


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
        if root.role != PrimitiveRole.UNIVERSE:
            raise ValueError(
                "Root Primitive must have role=PrimitiveRole.UNIVERSE. "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )

        segment_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]
        residue_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.RESIDUE]
        if not segment_nodes:
            raise ValueError("No SEGMENT-role Primitives found in hierarchy.")
        if not residue_nodes:
            raise ValueError("No RESIDUE-role Primitives found in hierarchy.")

        for leaf in root.leaves:
            if leaf.role != PrimitiveRole.PARTICLE:
                raise ValueError("All leaves must have role=PrimitiveRole.PARTICLE.")
            if leaf.element is None:
                raise ValueError(
                    f"Leaf Primitive '{leaf}' has role=PARTICLE but no element assigned. "
                    "AllAtomRDKitExportStrategy requires atomic PARTICLE leaves."
                )

        for seg in segment_nodes:
            nested_segs = [
                node for node in PreOrderIter(seg)
                if node.role == PrimitiveRole.SEGMENT and node is not seg
            ]
            if nested_segs:
                raise ValueError(
                    f"SEGMENT '{seg.label}' contains nested SEGMENT(s) "
                    f"{[n.label for n in nested_segs]}."
                )

        for res in residue_nodes:
            nested_res = [
                node for node in PreOrderIter(res)
                if node.role == PrimitiveRole.RESIDUE and node is not res
            ]
            if nested_res:
                raise ValueError(
                    f"RESIDUE '{res.label}' contains nested RESIDUE(s) "
                    f"{[n.label for n in nested_res]}."
                )

    def collect_mols(self, root: Primitive, resname_map: dict[str, str]) -> list[RDKitMolData]:
        """Walk the hierarchy and collect one RDKit topology per segment."""
        self.validate(root)

        mols_data: list[RDKitMolData] = []
        for segment in [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]:
            data = RDKitMolData(segment=segment)
            atom_id_to_local: dict[int, int] = {}
            mapped_atom_ids: set[int] = set()
            resid_counter = 1

            for residue in [node for node in PreOrderIter(segment) if node.role == PrimitiveRole.RESIDUE]:
                resname = _pdb_resname(residue.label, resname_map)
                particles = [
                    node for node in PreOrderIter(residue)
                    if node.is_leaf and node.role == PrimitiveRole.PARTICLE
                ]
                for atom in particles:
                    atom_id_to_local[id(atom)] = len(data.atoms)
                    mapped_atom_ids.add(id(atom))
                    data.atoms.append(atom)
                    if atom.shape is not None:
                        data.atom_positions.append(atom.shape.centroid)
                    else:
                        data.atom_positions.append(self.default_atom_position)
                    data.atom_resnames.append(resname)
                    data.atom_residue_labels.append(str(residue.label))
                    data.atom_particle_labels.append(str(atom.label))
                    data.atom_resids.append(resid_counter)
                resid_counter += 1

            segment_particles = [
                node for node in PreOrderIter(segment)
                if node.is_leaf and node.role == PrimitiveRole.PARTICLE
            ]
            if {id(atom) for atom in segment_particles} != mapped_atom_ids:
                raise ValueError(
                    f"SEGMENT '{segment.label}' contains PARTICLE leaves that are not "
                    "mapped through RESIDUE-role nodes."
                )

            bonds_set: set[tuple[int, int]] = set()
            for node in PreOrderIter(segment):
                if node.is_leaf or not node.internal_connections:
                    continue

                for conn_ref_pair in node.internal_connections:
                    conn_ref1, conn_ref2 = sorted(
                        conn_ref_pair,
                        key=lambda cr: (cr.primitive_handle, cr.connector_handle),
                    )
                    atom1 = _resolve_to_atom(node, conn_ref1)
                    atom2 = _resolve_to_atom(node, conn_ref2)
                    idx1 = atom_id_to_local[id(atom1)]
                    idx2 = atom_id_to_local[id(atom2)]
                    bond_pair = tuple(sorted((idx1, idx2)))
                    if bond_pair in bonds_set:
                        continue

                    data.bonds.append(bond_pair)
                    data.bond_refs.append((node, conn_ref1))
                    bonds_set.add(bond_pair)

            if data.bonds:
                sorted_bonds = sorted(
                    zip(data.bonds, data.bond_refs), key=lambda pair: pair[0]
                )
                data.bonds = [bond for bond, _ in sorted_bonds]
                data.bond_refs = [bond_ref for _, bond_ref in sorted_bonds]

            mols_data.append(data)

        return mols_data
