"""Shared topology traversal helpers for exporter interfaces."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from collections.abc import Hashable

from ...chemistry.core import BOND_ORDER
from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive


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
