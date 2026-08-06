"""Shared topology traversal helpers for exporter interfaces."""


from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Callable, Union

import networkx as nx
from anytree import PreOrderIter

from ...chemistry.core import BOND_ORDER
from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive
from ...roles import PrimitiveRole

# Type hints for new objects introduced
# ResolutionPredicate answers "should descent stop at this Primitive?".
# It is the same signature anytree expects for the "stop" argument of its
# iterators (PreOrderIter, LevelOrderIter, etc)
# ResolutionSpec is the user-facing form and resolution_predicate() normalizes every
# accepted form into a single ResolutionPredicate.
ResolutionPredicate = Callable[[Primitive], bool]
ResolutionSpec = Union[None, int, PrimitiveRole, ResolutionPredicate]


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


def _pdb_resname(
    label: Hashable,
    resname_map: dict[str, str],
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Map residue metadata or label to a PDB-compliant 3-character name."""
    label = str(label)
    if metadata is not None and "residue_name" in metadata:
        name = str(metadata["residue_name"])
    elif resname_map and label in resname_map:
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

#TODO: Maybe this can replace the logic in _resolve_to_atom? 2 possible routes
# 1) Keep _resolve_to_atom as a wrapper that calls _resolve_to_resolution
# 2) replace _resolve_to_atom entirely, and use _resolve_to_resolution
#       with the correct stop predicate anywhere _resolve_to_atom() is called 
def _resolve_to_resolution(
    parent: Primitive,
    conn_ref: ConnectorReference,
    resolution_ids: set[int],
    max_hops: int = 50
) -> Primitive:
    """Follow a connector reference down until it lands on a resolution node.

    Generalizes _resolve_to_atom: stop at "is a resolution node" instead of "is an
    atom". Only called from nodes above the floor.
    """
    for _ in range(max_hops):
        try:
            child = parent.fetch_child(conn_ref.primitive_handle)
        except (KeyError, AttributeError) as exc:
            raise ValueError(
                f"Cannot resolve to resolution node: child '{conn_ref.primitive_handle}' "
                f"not found under parent '{parent.label}'."
            ) from exc
        if id(child) in resolution_ids or child.is_leaf:
            return child
        try:
            conn_ref = child.external_connectors[conn_ref.connector_handle]
        except KeyError as exc:
            raise ValueError(
                f"Cannot resolve to resolution node: external connector "
                f"'{conn_ref.connector_handle}' not found on child '{child.label}' "
                f"(parent '{parent.label}')."
            ) from exc
        parent = child
    raise ValueError(
        f"_resolve_to_resolution exceeded {max_hops} hops; "
        "likely a malformed hierarchy or non-terminating connector references."
    )


def resolution_graph(root: Primitive, resolution: ResolutionSpec = None) -> nx.Graph:
    """Build the flat resolution graph for a given resolution floor.

    This method is useful for interfacing with other software packages, or file writers.

    A resolution designates a stopping floor through the hierarchy. Descending from
    a root, a node either stops (it becomes one resolution node) or is descended
    into; everything below a resolution node is collapsed into it.

    Nodes are integer ids (preorder over the floor), each carrying:
        primitive : the originating Primitive
        name      : element symbol for atoms, else the Primitive label
        element   : element symbol for atoms, else None
    Edges carry:
        bond_order: numeric bond order inferred from the crossing connector
    """
    should_stop = resolution_predicate(resolution, root)
    resolution_nodes = collect_resolution_nodes(root, should_stop)
    resolution_ids = {id(prim) for prim in resolution_nodes}
    index_of = {id(prim): idx for idx, prim in enumerate(resolution_nodes)}

    # Build flat graph, nodes are the floor primitives found above
    graph = nx.Graph()
    for idx, prim in enumerate(resolution_nodes):
        graph.add_node(
            idx,
            primitive=prim,
            name=prim.element.symbol if prim.is_atom else str(prim.label),
            element=prim.element.symbol if prim.is_atom else None,
        )
    # Build edges (bonds) in the flat graph
    # Descent stops just above the resolution floor, so nodes at and below it are never visited
    # This is done because a node's own sibling-level bonds are recorded at its parent level.
    # DEVNOTE: Possibly add a sibling_connections() method to Primitive?
    # This could simplify workflows like this where we have a set of nodes and want their sibling bonds.
    seen = set()
    for node in PreOrderIter(root, stop=should_stop):
        for conn_ref_pair in node.internal_connections:
            ref1, ref2 = sorted(conn_ref_pair, key=connector_reference_sort_key)
            end1 = _resolve_to_resolution(node, ref1, resolution_ids)
            end2 = _resolve_to_resolution(node, ref2, resolution_ids)
            if end1 is end2:
                continue
            idx1, idx2 = index_of[id(end1)], index_of[id(end2)]
            key = frozenset((idx1, idx2))
            if key in seen:
                continue
            seen.add(key)
            graph.add_edge(idx1, idx2, bond_order=_bond_order_from_conn_ref(node, ref1))

    return graph


def resolution_predicate(resolution: ResolutionSpec, root: Primitive) -> ResolutionPredicate:
    """Normalize any accepted resolution designation into one stop predicate.
    
    Stop predicates can include the following:
        None: full recursion into the leaves (fully atomistic / finest stored)
        int: a depth cut, measured relative to the passed root primitive
        PrimitiveRole: a role cut (SAAMR levels)
        Callable[[Primitive], bool]: a custom stop predicate

    A ResolutionPredicate is usable with anytree iterator's ``stop`` parameter.

    """
    if resolution is None:
        return lambda prim: False
    if isinstance(resolution, PrimitiveRole):
        if not any(node.role is resolution for node in PreOrderIter(root)):
            raise ValueError(f"No Primitive in the hierarchy has role {resolution}.")
        return lambda prim: prim.role is resolution
    # bool is an int subclass, reject it before the int case catches it
    if isinstance(resolution, bool):
        raise TypeError("resolution cannot be a bool.")
    if isinstance(resolution, int):
        base_depth = root.depth
        return lambda prim: (prim.depth - base_depth) >= resolution
    if callable(resolution):
        return resolution
    raise TypeError(
        f"Cannot interpret resolution of type {type(resolution).__name__}. "
        "Pass None, an int depth, a PrimitiveRole, or a callable stop predicate."
    )


def collect_resolution_nodes(root: Primitive, should_stop: ResolutionPredicate) -> list[Primitive]:
    """A recusrvie tree walk building a preordered list of floor nodes for a given ResolutionPredicate.

    Nodes are collected based on the ``should_stop`` predicate conditions or by .is_leaf().
    See ``resolution_predicate``.

    Note
    ----
    This operates differently than anytree's iterators. Here, the nodes
    triggering ``should_stop`` are captured. anytree's iterators exclude
    nodes triggering ``should_stop``.

    """
    if root.is_leaf or should_stop(root):
        return [root]
    return [
        node
        for child in root.children
        for node in collect_resolution_nodes(child, should_stop)
    ]
