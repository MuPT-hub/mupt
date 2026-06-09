"""Shared topology traversal helpers for exporter interfaces."""


from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field

from ...chemistry.core import BOND_ORDER
from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive
from ...roles import PrimitiveRole


@dataclass
class SAAMRRoleTopologyIndex:
    """Role-indexed view of a SAAMR-like Primitive hierarchy."""

    segments: list[Primitive] = field(default_factory=list)
    residues_by_segment: dict[int, list[Primitive]] = field(default_factory=dict)
    particles_by_residue: dict[int, list[Primitive]] = field(default_factory=dict)
    segment_of_node: dict[int, Primitive] = field(default_factory=dict)
    bond_nodes: list[Primitive] = field(default_factory=list)
    bond_nodes_by_segment: dict[int, list[Primitive]] = field(default_factory=dict)


@dataclass(frozen=True)
class SAAMRResidueRecord:
    """One RESIDUE-role node and its role-aware traversal context."""

    segment_idx: int
    segment: Primitive
    residue_idx: int
    residue_global_idx: int
    residue: Primitive
    particles: tuple[Primitive, ...]


def build_saamr_role_topology_index(root: Primitive) -> SAAMRRoleTopologyIndex:
    """Build a single-pass role index for a SAAMR-like Primitive hierarchy.

    The accepted hierarchy is role based rather than depth based: UNASSIGNED
    grouping nodes may appear between canonical SAAMR roles, but SEGMENT and
    RESIDUE roles cannot nest within the same role.
    """
    if root.role != PrimitiveRole.UNIVERSE:
        raise ValueError(
            "Root Primitive must have role=PrimitiveRole.UNIVERSE. "
            "Assign roles via assign_SAAMR_roles() or set them manually."
        )

    index = SAAMRRoleTopologyIndex()

    def visit(
        node: Primitive,
        current_segment: Primitive | None,
        current_residue: Primitive | None,
    ) -> None:
        role = node.role

        if role == PrimitiveRole.SEGMENT:
            if current_segment is not None:
                raise ValueError(
                    f"SEGMENT '{current_segment.label}' contains nested SEGMENT(s) "
                    f"['{node.label}']."
                )
            current_segment = node
            index.segments.append(node)
            index.residues_by_segment[id(node)] = []
            index.bond_nodes_by_segment[id(node)] = []

        elif role == PrimitiveRole.RESIDUE:
            if current_residue is not None:
                raise ValueError(
                    f"RESIDUE '{current_residue.label}' contains nested RESIDUE(s) "
                    f"['{node.label}']."
                )
            if current_segment is None:
                raise ValueError("RESIDUE-role Primitives must be enclosed by a SEGMENT.")
            current_residue = node
            index.residues_by_segment[id(current_segment)].append(node)
            index.particles_by_residue[id(node)] = []

        elif role == PrimitiveRole.PARTICLE and not node.is_leaf:
            raise ValueError("PARTICLE-role Primitives must be leaves.")

        if node.internal_connections:
            if current_segment is None:
                raise ValueError(
                    "SAAMR role-aware export does not support internal connections "
                    "owned above a SEGMENT-role Primitive. Move the bond owner into "
                    "a SEGMENT or represent the cross-segment relationship as external "
                    "connectors."
                )
            index.bond_nodes.append(node)
            index.bond_nodes_by_segment[id(current_segment)].append(node)

        if current_segment is not None:
            index.segment_of_node[id(node)] = current_segment

        if node.is_leaf:
            if role == PrimitiveRole.SEGMENT:
                raise ValueError(
                    f"SEGMENT-role Primitive '{node.label}' contains no RESIDUE-role descendants."
                )
            if role == PrimitiveRole.RESIDUE:
                raise ValueError(
                    f"RESIDUE-role Primitive '{node.label}' contains no PARTICLE leaves."
                )
            if role != PrimitiveRole.PARTICLE:
                raise ValueError("All leaves must have role=PrimitiveRole.PARTICLE.")
            if node.element is None:
                raise ValueError(
                    f"Leaf Primitive '{node}' has role=PARTICLE but no element assigned. "
                    "All-atom export requires atomic PARTICLE leaves."
                )
            if current_segment is None or current_residue is None:
                raise ValueError(
                    "PARTICLE leaves must be enclosed by RESIDUE and SEGMENT roles."
                )
            index.particles_by_residue[id(current_residue)].append(node)
            return

        for child in node.children:
            visit(child, current_segment, current_residue)

    visit(root, current_segment=None, current_residue=None)

    if not index.segments:
        raise ValueError("No SEGMENT-role Primitives found in hierarchy.")
    if not index.particles_by_residue:
        raise ValueError("No RESIDUE-role Primitives found in hierarchy.")
    for segment in index.segments:
        residues = index.residues_by_segment[id(segment)]
        if not residues:
            raise ValueError(
                f"SEGMENT-role Primitive '{segment.label}' contains no RESIDUE-role descendants."
            )
        empty_residues = [
            residue.label
            for residue in residues
            if not index.particles_by_residue[id(residue)]
        ]
        if empty_residues:
            raise ValueError(
                f"SEGMENT-role Primitive '{segment.label}' contains RESIDUE-role "
                f"Primitive(s) with no PARTICLE leaves: {empty_residues}."
            )

    return index


def iter_saamr_residue_records(
    index: SAAMRRoleTopologyIndex,
) -> Iterator[SAAMRResidueRecord]:
    """Yield RESIDUE-role records in deterministic SAAMR traversal order."""
    residue_global_idx = 0
    for segment_idx, segment in enumerate(index.segments):
        for residue_idx, residue in enumerate(
            index.residues_by_segment[id(segment)],
            start=1,
        ):
            yield SAAMRResidueRecord(
                segment_idx=segment_idx,
                segment=segment,
                residue_idx=residue_idx,
                residue_global_idx=residue_global_idx,
                residue=residue,
                particles=tuple(index.particles_by_residue[id(residue)]),
            )
            residue_global_idx += 1


def _pdb_resname(label: Hashable, resname_map: dict[str, str]) -> str:
    """Map a residue label to a PDB-compliant 3-character residue name."""
    label = str(label)
    if resname_map and label in resname_map:
        name = resname_map[label]
    else:
        name = label

    if len(name) != 3:
        raise ValueError(f"Residue name '{name}' (from '{label}') is not 3 characters long")
    return name.upper()


def connector_reference_sort_key(conn_ref: ConnectorReference) -> tuple[str, str]:
    """Return a deterministic key for connector refs with arbitrary hashable handles."""
    return (repr(conn_ref.primitive_handle), repr(conn_ref.connector_handle))


def _resolve_to_atom(
    parent: Primitive,
    conn_ref: ConnectorReference,
    _depth: int = 0,
    _max_depth: int = 50,
) -> Primitive:
    """Recursively follow external connectors to find the leaf atom."""
    if _depth > _max_depth:
        raise ValueError(
            f"_resolve_to_atom exceeded maximum recursion depth ({_max_depth}) "
            f"starting from parent '{parent.label}' at connector "
            f"({conn_ref.primitive_handle}, {conn_ref.connector_handle}). "
            "This indicates non-terminating connector resolution, likely caused by "
            "malformed hierarchy or connector references."
        )

    try:
        child = parent.fetch_child(conn_ref.primitive_handle)
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"Cannot resolve atom: child '{conn_ref.primitive_handle}' "
            f"not found under parent '{parent.label}'."
        ) from exc

    if child.is_atom:
        return child

    try:
        next_ref = child.external_connectors[conn_ref.connector_handle]
    except KeyError as exc:
        raise ValueError(
            f"Cannot resolve atom: external connector "
            f"'{conn_ref.connector_handle}' not found on child "
            f"'{child.label}' (parent '{parent.label}'). "
            "Ensure the primitive tree has well-formed connector chains."
        ) from exc

    return _resolve_to_atom(child, next_ref, _depth=_depth + 1, _max_depth=_max_depth)


def resolve_to_atom_cached(
    parent: Primitive,
    conn_ref: ConnectorReference,
    cache: dict[tuple[int, object, object], Primitive],
) -> Primitive:
    """Resolve a connector reference to an atom using a caller-owned cache."""
    cache_key = (id(parent), conn_ref.primitive_handle, conn_ref.connector_handle)
    if cache_key not in cache:
        cache[cache_key] = _resolve_to_atom(parent, conn_ref)
    return cache[cache_key]


def _bond_order_from_conn_ref(parent: Primitive, conn_ref: ConnectorReference) -> float:
    """Infer numeric bond order from a connection reference."""
    connector = parent.fetch_connector_on_child(conn_ref)
    return BOND_ORDER[connector.bondtype]
