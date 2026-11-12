"""Tests for SAAMR depth and role helpers on (new-style) Primitive hierarchies"""

import pytest

from mupt.chemistry import ELEMENTS
from mupt.mupr.primitives import AtomicPrimitive, CompositePrimitive, RootPrimitive
from mupt.mupr.primitives.predicates import has_strict_SAAMR_depth
from mupt.mupr.primitives.roles import (
    PrimitiveRole,
    assign_SAAMR_roles,
    has_SAAMR_roles,
)


def _strict_tree() -> RootPrimitive:
    """Strict SAAMR tree: universe -> chain -> residue -> He atom at depth 3"""
    universe = RootPrimitive(label="universe")
    chain = CompositePrimitive(label="chain")
    unit = CompositePrimitive(label="HEL")
    universe.attach_child(chain)
    chain.attach_child(unit)
    unit.attach_child(AtomicPrimitive(element=ELEMENTS[2], label="He"))
    return universe


def _depth4_tree() -> RootPrimitive:
    """Same as the strict tree, with an extra grouping level above the chain"""
    universe = RootPrimitive(label="universe", role=PrimitiveRole.UNIVERSE)
    domain = CompositePrimitive(label="domain")
    chain = CompositePrimitive(label="chain", role=PrimitiveRole.SEGMENT)
    unit = CompositePrimitive(label="HEL", role=PrimitiveRole.RESIDUE)
    atom = AtomicPrimitive(element=ELEMENTS[2], label="He", role=PrimitiveRole.PARTICLE)
    universe.attach_child(domain)
    domain.attach_child(chain)
    chain.attach_child(unit)
    unit.attach_child(atom)
    return universe


def test_strict_depth_detected() -> None:
    """A universe/segment/residue/atom tree has strict SAAMR depth"""
    assert has_strict_SAAMR_depth(_strict_tree())


def test_extra_level_is_not_strict_depth() -> None:
    """An extra grouping level puts atoms at depth 4"""
    assert not has_strict_SAAMR_depth(_depth4_tree())


def test_non_atom_leaf_is_not_strict_depth() -> None:
    """Leaves at depth 3 must be atoms"""
    universe = RootPrimitive(label="universe")
    chain = CompositePrimitive(label="chain")
    unit = CompositePrimitive(label="unit")
    universe.attach_child(chain)
    chain.attach_child(unit)
    unit.attach_child(CompositePrimitive(label="bead"))
    assert not has_strict_SAAMR_depth(universe)


def test_assign_then_has_roles() -> None:
    """assign_SAAMR_roles labels every level, after which has_SAAMR_roles is True"""
    universe = _strict_tree()
    assert not has_SAAMR_roles(universe)
    assign_SAAMR_roles(universe)
    assert has_SAAMR_roles(universe)
    chain = universe.children[0]
    unit = chain.children[0]
    atom = unit.children[0]
    assert [p.role for p in (universe, chain, unit, atom)] == [
        "universe",
        "segment",
        "residue",
        "particle",
    ]


def test_assign_refuses_non_strict_tree() -> None:
    """assign_SAAMR_roles raises rather than guessing on a depth-4 tree"""
    with pytest.raises(ValueError):
        assign_SAAMR_roles(_depth4_tree())


def test_hand_assigned_roles_detected_with_unassigned_level() -> None:
    """Roles set by hand count, and an UNASSIGNED grouping level is fine"""
    assert has_SAAMR_roles(_depth4_tree())


def test_custom_role_strings_are_ignored() -> None:
    """Custom (non-SAAMR) role strings neither raise nor count as SAAMR roles"""
    universe = _strict_tree()
    universe.role = "box"
    assert not has_SAAMR_roles(universe)
