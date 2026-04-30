"""Shared topology traversal helpers for exporter interfaces."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from collections.abc import Hashable
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


def build_saamr_role_topology_index(root: Primitive) -> SAAMRRoleTopologyIndex:
    """Build a single-pass role index for a SAAMR-like Primitive hierarchy."""
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

        # Carry the active SEGMENT/RESIDUE context through unassigned grouping nodes.
        if role == PrimitiveRole.SEGMENT:
            if current_segment is not None:
                raise ValueError(
                    f"SEGMENT '{current_segment.label}' contains nested SEGMENT(s) "
                    f"['{node.label}']."
                )
            current_segment = node
            index.segments.append(node)
            index.residues_by_segment[id(node)] = []

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

        # Index bond-owning nodes once so exporters do not rescan every segment subtree.
        if current_segment is not None:
            index.segment_of_node[id(node)] = current_segment
            if node.internal_connections:
                index.bond_nodes.append(node)

        if node.is_leaf:
            # A valid all-atom SAAMR leaf must be an atomic PARTICLE inside both roles.
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

        # Explicit recursion keeps this a single DFS while preserving role context.
        for child in node.children:
            visit(child, current_segment, current_residue)

    visit(root, current_segment=None, current_residue=None)

    if not index.segments:
        raise ValueError("No SEGMENT-role Primitives found in hierarchy.")
    if not index.particles_by_residue:
        raise ValueError("No RESIDUE-role Primitives found in hierarchy.")

    return index


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


def _resolve_to_atom(
    parent: Primitive,
    conn_ref: ConnectorReference,
    _depth: int = 0,
    _max_depth: int = 50,
) -> Primitive:
    """Recursively follow external_connectors to find the leaf atom."""
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

    # External connectors mirror the same connector handle down to the next level.
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


def _bond_order_from_conn_ref(parent: Primitive, conn_ref: ConnectorReference) -> float:
    """Infer numeric bond order from a connection reference at the current parent level."""
    connector = parent.fetch_connector_on_child(conn_ref)
    return BOND_ORDER[connector.bondtype]
