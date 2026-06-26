"""Tests for shared SAAMR role topology traversal helpers."""


import pytest

from mupt.chemistry import ELEMENTS
from mupt.interfaces._shared.topology import _pdb_resname, build_saamr_role_topology_index
from mupt.mupr.primitives import Primitive
from mupt.roles import PrimitiveRole


def _particle(label: str) -> Primitive:
    return Primitive(label=label, element=ELEMENTS[1], role=PrimitiveRole.PARTICLE)


def test_build_saamr_role_topology_index_allows_unassigned_grouping_nodes():
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    group = Primitive(label="group")
    segment = Primitive(label="segment", role=PrimitiveRole.SEGMENT)
    residue = Primitive(label="residue", role=PrimitiveRole.RESIDUE)
    atom = _particle("H")

    universe.attach_child(group)
    group.attach_child(segment)
    segment.attach_child(residue)
    residue.attach_child(atom)

    index = build_saamr_role_topology_index(universe)

    assert index.segments == [segment]
    assert index.residues_by_segment[id(segment)] == [residue]
    assert index.particles_by_residue[id(residue)] == [atom]
    assert index.segment_of_node[id(atom)] is segment


def test_build_saamr_role_topology_index_rejects_empty_segment():
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    universe.attach_child(Primitive(label="empty", role=PrimitiveRole.SEGMENT))

    with pytest.raises(ValueError, match="contains no RESIDUE"):
        build_saamr_role_topology_index(universe)


def test_build_saamr_role_topology_index_rejects_nested_residue():
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    segment = Primitive(label="segment", role=PrimitiveRole.SEGMENT)
    residue = Primitive(label="residue", role=PrimitiveRole.RESIDUE)
    nested = Primitive(label="nested", role=PrimitiveRole.RESIDUE)
    atom = _particle("H")

    universe.attach_child(segment)
    segment.attach_child(residue)
    residue.attach_child(nested)
    nested.attach_child(atom)

    with pytest.raises(ValueError, match="nested RESIDUE"):
        build_saamr_role_topology_index(universe)


def test_pdb_resname_prefers_residue_metadata_for_instance_labels():
    resname = _pdb_resname(
        "head_styrene_000",
        {"head_styrene": "PSH"},
        metadata={"residue_name": "PSH"},
    )

    assert resname == "PSH"
