"""Strategy implementations for MuPT -> MDAnalysis export."""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from networkx import Graph

from ...chemistry.core import BOND_ORDER_ATTR
from ...mupr.primitives.primitive import Primitive
from ...mupr.primitives.predicates import is_atom
from ...roles import PrimitiveRole
from ...trees.subselect import NodePredicate, primoprogenitors


@dataclass
class MDATopologyData:
    """Container for topology arrays/lists used to build an MDAnalysis Universe."""

    atom_elements: list[str] = field(default_factory=list)
    atom_names: list[str] = field(default_factory=list)
    atom_types: list[str] = field(default_factory=list)
    atom_masses: list[float] = field(default_factory=list)
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


def _pdb_resname(
    label: Hashable,
    resname_map: Optional[Mapping[str, str]],
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    """Map residue metadata or label to a PDB-compliant 3-character name.

    Precedence: ``metadata["residue_name"]``, then ``resname_map[label]``,
    then the label itself.
    """
    label = str(label)
    if metadata is not None and "residue_name" in metadata:
        name = str(metadata["residue_name"])
    elif resname_map and label in resname_map:
        name = resname_map[label]
    else:
        name = label

    if len(name) != 3:
        raise ValueError(
            f"Residue name '{name}' (from '{label}') is not 3 characters long"
        )
    return name.upper()


def role_of(prim: Primitive) -> Optional[str]:
    """The role of a Primitive as a lowercase string, or None if it has none.

    Accepts either a ``PrimitiveRole`` member or its string value
    (``"segment"``, ``"residue"``, ...).
    """
    # TODO: read the role field directly once roles are restored on the new
    # Primitive classes (#151, #162); until then this reads an ad-hoc attribute
    role = getattr(prim, "role", None)
    if isinstance(role, PrimitiveRole):
        role = role.value
    return role.lower() if isinstance(role, str) else None


def has_role(role: PrimitiveRole) -> NodePredicate[Primitive]:
    """Predicate for whether a Primitive carries the given role."""

    def predicate(prim: Primitive) -> bool:
        return role_of(prim) == role.value

    return predicate


@dataclass
class _ResidueRecord:
    """One exported residue: its Primitive, its segment and its particles."""

    segment_idx: int
    residue_id: int  # 1-based within its segment
    residue: Primitive
    particles: list[Primitive]


def _reject_nested(prims: list[Primitive], role: PrimitiveRole) -> None:
    """Raise if any of these Primitives has a descendant with the same role."""
    for prim in prims:
        for descendant in prim.descendants:
            if role_of(descendant) == role.value:
                raise ValueError(
                    f"'{role.value}'-role Primitive '{prim.label}' contains another "
                    f"'{role.value}'-role Primitive '{descendant.label}'."
                )


def _collect_residue_records(
    root: Primitive,
    particle_selector: NodePredicate[Primitive],
) -> tuple[list[Primitive], list[_ResidueRecord]]:
    """Walk root -> segments -> residues -> particles, keeping each residue contiguous.

    Segments and residues are the Primitives with role ``"segment"`` and
    ``"residue"``; they may sit at any depth, with untagged grouping
    Primitives in between. Particles are the first Primitives below each
    residue that satisfy ``particle_selector``.

    The walk is done level by level, rather than taking particles in the order
    of ``root.cross_section(particle_selector)``, because the cross-section
    yields particles in breadth-first order: in hierarchies where particles sit
    at different depths, that order interleaves particles from different
    residues, which file formats such as PDB and GRO cannot represent.
    """
    segment_role, residue_role = PrimitiveRole.SEGMENT, PrimitiveRole.RESIDUE

    segments = list(primoprogenitors(root, has_role(segment_role)))
    if not segments:
        raise ValueError(
            "No Primitive has role 'segment'. Tag each chain or molecule, "
            "e.g. chain.role = 'segment', and each repeat unit, "
            "e.g. unit.role = 'residue'."
        )
    _reject_nested(segments, segment_role)

    records: list[_ResidueRecord] = []
    for segment_idx, segment in enumerate(segments):
        residues = list(primoprogenitors(segment, has_role(residue_role)))
        if not residues:
            raise ValueError(
                f"Segment '{segment.label}' contains no Primitive with role 'residue'."
            )
        _reject_nested(residues, residue_role)

        for residue_id, residue in enumerate(residues, start=1):
            particles = list(primoprogenitors(residue, particle_selector))
            if not particles:
                raise ValueError(
                    f"Residue '{residue.label}' in segment '{segment.label}' "
                    "contains no particles to export."
                )
            records.append(_ResidueRecord(segment_idx, residue_id, residue, particles))

    n_assigned = sum(len(record.particles) for record in records)
    n_total = sum(1 for _ in primoprogenitors(root, particle_selector))
    if n_assigned != n_total:
        raise ValueError(
            f"{n_total - n_assigned} of {n_total} particles are not inside both a "
            "'segment'-role and a 'residue'-role Primitive."
        )

    return segments, records


def _add_bonds(
    data: MDATopologyData,
    cross_section: Graph,
    index_of: dict[int, int],
) -> None:
    """Add deduplicated, sorted bonds (and bond orders) from a cross-section graph."""
    bonds: dict[tuple[int, int], float] = {}
    for prim_a, prim_b, edge_data in cross_section.edges(data=True):
        pair = tuple(sorted((index_of[id(prim_a)], index_of[id(prim_b)])))
        # duplicate edges (multigraph) between the same pair are collapsed;
        # MuPT enforces matching bond orders on both Connectors of a bond
        bonds.setdefault(pair, edge_data.get(BOND_ORDER_ATTR))

    for pair in sorted(bonds):
        data.bonds.append(pair)
        data.bonds_set.add(pair)
        data.bond_orders.append(bonds[pair])

    if any(order is None for order in data.bond_orders):
        data.bond_orders = []  # export connectivity without orders


def _position_of(prim: Primitive) -> Optional[np.ndarray]:
    """Position of a Primitive: its atoms' center of mass, else its own shape.

    For an atom this is just the atom's position. Returns None if no position
    can be determined.
    """
    atoms = list(primoprogenitors(prim, is_atom))
    if atoms and all(atom.shape is not None for atom in atoms):
        positions = np.array([atom.shape.centroid for atom in atoms], dtype=float)
        masses = np.array([atom.mass for atom in atoms], dtype=float)
        return (masses[:, None] * positions).sum(axis=0) / masses.sum()

    if prim.shape is not None:
        return np.asarray(prim.shape.centroid, dtype=float)

    return None


class MDAExportStrategy(ABC):
    """Abstract strategy for collecting MDAnalysis-exportable topology data."""

    @abstractmethod
    def collect_topology(
        self, root: Primitive, resname_map: Mapping[str, str]
    ) -> MDATopologyData:
        """Collect topology attributes from a Primitive hierarchy."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name for this strategy."""


class AllAtomExportStrategy(MDAExportStrategy):
    """All-atom export: one MDAnalysis atom per AtomicPrimitive.

    Segments and residues are the Primitives with role ``"segment"`` and
    ``"residue"``, at any depth. Atoms, bonds and bond orders come from the
    atom cross-section of the hierarchy (``root.cross_section(is_atom)``);
    atoms are written segment by segment and residue by residue.
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
                raise ValueError("default_atom_position must be a 3-dimensional vector")
            self.default_atom_position = default_atom_position

    @property
    def label(self) -> str:
        """Human-readable strategy name."""
        return "All-atom"

    def collect_topology(
        self, root: Primitive, resname_map: Mapping[str, str]
    ) -> MDATopologyData:
        """Walk the hierarchy and gather MDAnalysis topology arrays/lists."""
        segments, records = _collect_residue_records(root, is_atom)

        data = MDATopologyData(num_segments=len(segments))
        index_of: dict[int, int] = {}
        for residue_global_idx, record in enumerate(records):
            data.residue_names.append(
                _pdb_resname(
                    record.residue.label, resname_map, metadata=record.residue.metadata
                )
            )
            data.residue_segindex.append(record.segment_idx)
            data.residue_ids.append(record.residue_id)

            for atom in record.particles:
                index_of[id(atom)] = len(data.atom_elements)
                symbol = atom.element.symbol
                data.atom_elements.append(symbol)
                data.atom_names.append(symbol)
                data.atom_types.append(symbol)
                data.atom_masses.append(atom.mass)
                data.atom_resindex.append(residue_global_idx)
                data.atom_segindex.append(record.segment_idx)

                position = _position_of(atom)
                data.atom_positions.append(
                    list(self.default_atom_position if position is None else position)
                )

        _add_bonds(data, root.cross_section(is_atom), index_of)
        return data


class CoarseGrainedExportStrategy(MDAExportStrategy):
    """Coarse-grained export: one MDAnalysis bead (and residue) per residue.

    Each ``"residue"``-role Primitive becomes one bead, inside the
    ``"segment"``-role Primitive above it, i.e. one bead per repeat unit for
    Root -> chain -> repeat unit hierarchies, with or without atoms below the
    repeat units. Bonds between beads come from the bead cross-section of the
    hierarchy.

    Each bead is placed at the mass-weighted center of its atoms if they all
    have positions, otherwise at its own position if it has one, otherwise at
    ``default_bead_position``.
    Bead names and types are the bead's residue name.
    """

    def __init__(
        self,
        default_bead_position: Optional[np.ndarray] = None,
    ) -> None:
        if default_bead_position is None:
            self.default_bead_position = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            default_bead_position = np.asarray(default_bead_position, dtype=float)
            if default_bead_position.shape != (3,):
                raise ValueError("default_bead_position must be a 3-dimensional vector")
            self.default_bead_position = default_bead_position

    @property
    def label(self) -> str:
        """Human-readable strategy name."""
        return "Coarse-grained"

    def collect_topology(
        self, root: Primitive, resname_map: Mapping[str, str]
    ) -> MDATopologyData:
        """Walk the hierarchy and gather one bead per residue."""
        # each bead is its own residue, so the residue and particle levels coincide
        is_bead = has_role(PrimitiveRole.RESIDUE)
        segments, records = _collect_residue_records(root, is_bead)

        data = MDATopologyData(num_segments=len(segments))
        index_of: dict[int, int] = {}
        masses: list[Optional[float]] = []
        for residue_global_idx, record in enumerate(records):
            (bead,) = record.particles
            resname = _pdb_resname(bead.label, resname_map, metadata=bead.metadata)
            data.residue_names.append(resname)
            data.residue_segindex.append(record.segment_idx)
            data.residue_ids.append(record.residue_id)

            index_of[id(bead)] = len(data.atom_names)
            data.atom_names.append(resname)
            data.atom_types.append(resname)
            data.atom_resindex.append(residue_global_idx)
            data.atom_segindex.append(record.segment_idx)
            try:
                masses.append(bead.mass)
            except TypeError:  # some descendant has no mass
                masses.append(None)

            position = _position_of(bead)
            data.atom_positions.append(
                list(self.default_bead_position if position is None else position)
            )

        if all(mass is not None for mass in masses):
            data.atom_masses = masses

        _add_bonds(data, root.cross_section(is_bead), index_of)
        return data
