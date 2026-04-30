"""Strategy implementations for MuPT -> MDAnalysis export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Hashable
from typing import Optional

from anytree import PreOrderIter
import numpy as np

from ...chemistry.core import BOND_ORDER
from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive
from ...roles import PrimitiveRole


def _pdb_resname(label: Hashable, resname_map: dict[str, str]) -> str:
    """Map a residue label to a PDB-compliant 3-character residue name.

    Parameters
    ----------
    label : Hashable
        Original residue label from the Primitive object.  Non-string
        labels are normalized to ``str(label)`` before lookup and
        length validation.
    resname_map : dict[str, str]
        Mapping from residue labels to 3-character PDB residue names.

    Returns
    -------
    str
        Uppercase, 3-character PDB-compliant residue name.

    Raises
    ------
    ValueError
        If the resulting residue name is not exactly 3 characters long.
    """
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
    """Recursively follow external_connectors to find the leaf atom.

    Raises
    ------
    ValueError
        If the connector chain is broken (missing child or external
        connector entry) or if recursion exceeds *_max_depth*, which
        indicates non-terminating connector resolution, typically caused
        by malformed hierarchy or connector references.
    """
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
    """Infer bond order from a connection reference at the current parent level.

    Parameters
    ----------
    parent : Primitive
        The parent node whose ``internal_connections`` contain *conn_ref*.
    conn_ref : ConnectorReference
        Reference to the child primitive and its connector handle.

    Returns
    -------
    float
        Numeric bond order derived from the connector's bondtype.

    Raises
    ------
    MissingConnectorError
        If the referenced connector does not exist on the child primitive.
        This indicates a corrupted primitive tree (connectors are always
        present for valid internal connections).
    KeyError
        If the connector's bondtype is not in the BOND_ORDER lookup table.
    """
    connector = parent.fetch_connector_on_child(conn_ref)
    return BOND_ORDER[connector.bondtype]


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
    grouping between UNIVERSE and SEGMENT) are traversed transparently via
    ``PreOrderIter`` and carry ``PrimitiveRole.UNASSIGNED``.
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
        """Validate role assignments needed for all-atom export.

        Checks performed:
        - Root must have UNIVERSE role
        - At least one SEGMENT and one RESIDUE must exist
        - All leaves must have PARTICLE role
        - No SEGMENT may contain a descendant SEGMENT (nesting forbidden)
        - No RESIDUE may contain a descendant RESIDUE (nesting forbidden)
        """
        if root.role != PrimitiveRole.UNIVERSE:
            raise ValueError(
                "Root Primitive must have role=PrimitiveRole.UNIVERSE. "
                "(Unassigned nodes default to PrimitiveRole.UNASSIGNED.) "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )

        segment_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]
        residue_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.RESIDUE]
        if not segment_nodes:
            raise ValueError(
                "No SEGMENT-role Primitives found in hierarchy. "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )
        if not residue_nodes:
            raise ValueError(
                "No RESIDUE-role Primitives found in hierarchy. "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )

        for leaf in root.leaves:
            if leaf.role != PrimitiveRole.PARTICLE:
                raise ValueError(
                    "All leaves must have role=PrimitiveRole.PARTICLE. "
                    "Assign roles via assign_SAAMR_roles() or set them manually."
                )

        # Reject nested same-role nodes that would cause double-counting
        for seg in segment_nodes:
            nested_segs = [
                node for node in PreOrderIter(seg)
                if node.role == PrimitiveRole.SEGMENT and node is not seg
            ]
            if nested_segs:
                raise ValueError(
                    f"SEGMENT '{seg.label}' contains nested SEGMENT(s) "
                    f"{[n.label for n in nested_segs]}. "
                    "Nested SEGMENT roles are not allowed — each SEGMENT must "
                    "define a non-overlapping partition of the hierarchy."
                )

        for res in residue_nodes:
            nested_res = [
                node for node in PreOrderIter(res)
                if node.role == PrimitiveRole.RESIDUE and node is not res
            ]
            if nested_res:
                raise ValueError(
                    f"RESIDUE '{res.label}' contains nested RESIDUE(s) "
                    f"{[n.label for n in nested_res]}. "
                    "Nested RESIDUE roles are not allowed — each RESIDUE must "
                    "define a non-overlapping partition of its parent SEGMENT."
                )

    def collect_topology(self, root: Primitive, resname_map: dict[str, str]) -> MDATopologyData:
        """Walk the hierarchy and gather MDAnalysis topology arrays/lists."""
        self.validate(root)
        data = MDATopologyData()

        # Build global atom index from DFS pre-order leaf traversal
        leaves = list(root.leaves)
        n_atoms = len(leaves)
        atom_id_to_global: dict[int, int] = {id(atom): idx for idx, atom in enumerate(leaves)}

        # Pre-allocate resindex/segindex arrays so we can write by global index
        data.atom_resindex = [0] * n_atoms
        data.atom_segindex = [0] * n_atoms

        # Collect per-atom data in DFS leaf order
        for atom in leaves:
            if atom.element is None:
                raise ValueError(
                    f"Leaf Primitive '{atom}' has role=PARTICLE but no element assigned. "
                    "AllAtomExportStrategy requires atomic PARTICLE leaves."
                )
            atom_symbol = atom.element.symbol
            data.atom_elements.append(atom_symbol)
            data.atom_names.append(atom_symbol)

            if hasattr(atom, "shape") and atom.shape is not None:
                data.atom_positions.append(list(atom.shape.centroid))
            else:
                data.atom_positions.append(list(self.default_atom_position))

        # Walk segment -> residue -> particle to build hierarchy mapping
        # Use PreOrderIter to find SEGMENT nodes at any depth (not just direct children)
        segment_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]
        data.num_segments = len(segment_nodes)
        mapped_count = 0

        for seg_idx, segment in enumerate(segment_nodes):
            resid_counter = 1
            residue_nodes = [node for node in PreOrderIter(segment) if node.role == PrimitiveRole.RESIDUE]

            for residue in residue_nodes:
                data.residue_names.append(_pdb_resname(residue.label, resname_map))
                data.residue_segindex.append(seg_idx)
                data.residue_ids.append(resid_counter)

                res_global_idx = len(data.residue_names) - 1
                particle_leaves = [
                    node
                    for node in PreOrderIter(residue)
                    if node.is_leaf and node.role == PrimitiveRole.PARTICLE
                ]
                for atom in particle_leaves:
                    atom_global_idx = atom_id_to_global[id(atom)]
                    data.atom_resindex[atom_global_idx] = res_global_idx
                    data.atom_segindex[atom_global_idx] = seg_idx
                    mapped_count += 1

                resid_counter += 1

        if mapped_count != n_atoms:
            raise ValueError(
                f"Role hierarchy traversal mapped {mapped_count} of {n_atoms} PARTICLE leaves "
                "to a RESIDUE and SEGMENT. Assign roles via assign_SAAMR_roles() or set them manually."
            )

        for node in PreOrderIter(root):
            if node.is_leaf or not node.internal_connections:
                continue

            for conn_ref_pair in node.internal_connections:
                # sort pair order for standardized bond-order reference
                # frozenset iteration order is arbitrary, so we sort by handles
                ref_list = sorted(
                    conn_ref_pair,
                    key=lambda cr: (cr.primitive_handle, cr.connector_handle),
                )
                conn_ref1, conn_ref2 = ref_list
                atom1 = _resolve_to_atom(node, conn_ref1)
                atom2 = _resolve_to_atom(node, conn_ref2)

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
