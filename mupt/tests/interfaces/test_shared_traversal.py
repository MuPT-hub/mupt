"""
Tests for shared helpers which extract SAAMR roles
from traversing MuPT representation hierarchies
"""
# ruff: noqa: D103 ("undocumented public function")

import pytest

from mupt.chemistry import ELEMENTS
from mupt.interfaces._shared.traversal import (
    _pdb_resname,
    build_saamr_role_topology_index,
)
from mupt.mupr.primitives import RootPrimitive, CompositePrimitive, AtomicPrimitive
from mupt.roles import PrimitiveRole


def test_build_saamr_role_topology_index_allows_unassigned_grouping_nodes():
    universe = RootPrimitive(label="universe", role=PrimitiveRole.UNIVERSE)
    group = CompositePrimitive(label="group")
    segment = CompositePrimitive(label="segment", role=PrimitiveRole.SEGMENT)
    residue = CompositePrimitive(label="residue", role=PrimitiveRole.RESIDUE)
    atom = AtomicPrimitive(element=ELEMENTS[1], label="H", role=PrimitiveRole.PARTICLE)

    group.parent = universe
    segment.parent = group
    residue.parent = segment
    atom.parent = residue

    index = build_saamr_role_topology_index(universe)

    assert index.segments == [segment]
    assert index.residues_by_segment[id(segment)] == [residue]
    assert index.particles_by_residue[id(residue)] == [atom]
    assert index.segment_of_node[id(atom)] is segment


def test_build_saamr_role_topology_index_rejects_empty_segment():
    universe = RootPrimitive(label="universe", role=PrimitiveRole.UNIVERSE)
    empty = CompositePrimitive(label="empty", role=PrimitiveRole.SEGMENT)
    empty.parent = universe

    with pytest.raises(ValueError, match="contains no RESIDUE"):
        build_saamr_role_topology_index(universe)


def test_build_saamr_role_topology_index_rejects_nested_residue():
    universe = RootPrimitive(label="universe", role=PrimitiveRole.UNIVERSE)
    segment = CompositePrimitive(label="segment", role=PrimitiveRole.SEGMENT)
    residue = CompositePrimitive(label="residue", role=PrimitiveRole.RESIDUE)
    nested = CompositePrimitive(label="nested", role=PrimitiveRole.RESIDUE)
    atom = AtomicPrimitive(element=ELEMENTS[1], label="H", role=PrimitiveRole.PARTICLE)

    segment.parent = universe
    residue.parent = segment
    nested.parent = residue
    atom.parent = nested

    with pytest.raises(ValueError, match="nested RESIDUE"):
        build_saamr_role_topology_index(universe)


def test_pdb_resname_prefers_residue_metadata_for_instance_labels():
    resname = _pdb_resname(
        "head_styrene_000",
        {"head_styrene": "PSH"},
        metadata={"residue_name": "PSH"},
    )

    assert resname == "PSH"
