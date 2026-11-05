"""Unit tests for hieraarchy assembly and modification operations on MuPT Primitives"""

import pytest

from mupt.mupr.primitives.primitive import (
    Primitive,
    SupportsParents,
    SupportsChildren,
    ImproperHierarchyError,
    ArborescenceError,
)
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.simple.atomic import AtomicPrimitive

from mupt.mutils.iteration import sliding_window
from mupt.chemistry.core import ELEMENTS


# Fixtures
def dummy_hierarchy_atop_prim(prim: Primitive, num_intermed: int = 3) -> RootPrimitive:
    """
    Build a single-branch hierarchy which has
    the passed Primitive instance as its sole leaf
    """
    root = RootPrimitive()
    hierarchy_prims: list[Primitive] = [
        root,
        *[CompositePrimitive() for _ in range(num_intermed)],
        prim,
    ]
    for parent_prim, child_prim in sliding_window(hierarchy_prims, n=2):
        child_prim.parent = parent_prim

    return root


# Tests proper
## Hierarchy mutability permissions
@pytest.mark.parametrize(
    "prim",
    [
        # no Root; can't give it a parent
        CompositePrimitive(),
        SimplePrimitive(),
        AtomicPrimitive(ELEMENTS[2]),
    ],
)
def test_frozen_hierarchy_add_parent(prim: SupportsParents) -> None:
    """Test that hierarchy parent addition is blocked by freezing it on any Primitive"""
    parent = RootPrimitive()
    prim.freeze_hierarchy()
    with pytest.raises(AttributeError):
        prim.parent = parent


@pytest.mark.parametrize(
    "prim",
    [
        # no Root; can't give it a parent
        CompositePrimitive(),
        SimplePrimitive(),
        AtomicPrimitive(ELEMENTS[2]),
    ],
)
def test_frozen_hierarchy_remove_parent(prim: SupportsParents) -> None:
    """Test that hierarchy parent removal is blocked by freezing it on any Primitive"""
    parent = RootPrimitive()
    prim.unfreeze_hierarchy()
    prim.parent = parent
    prim.freeze_hierarchy()
    with pytest.raises(AttributeError):
        del prim.parent


@pytest.mark.parametrize(
    "prim",
    [
        RootPrimitive(),
        CompositePrimitive(),
        # No simples; can't give them children
    ],
)
def test_frozen_hierarchy_add_children(prim: SupportsChildren) -> None:
    """Test that hierarchy child addition is blocked by freezing it on any Primitive"""
    child = SimplePrimitive()
    prim.freeze_hierarchy()
    with pytest.raises(AttributeError):
        prim.children = [child]


@pytest.mark.parametrize(
    "prim",
    [
        RootPrimitive(),
        CompositePrimitive(),
        # No simples; can't give them children
    ],
)
def test_frozen_hierarchy_remove_children(prim: SupportsChildren) -> None:
    """Test that hierarchy child removal is blocked by freezing it on any Primitive"""
    child = SimplePrimitive()
    prim.unfreeze_hierarchy()
    prim.children = [child]
    prim.freeze_hierarchy()
    with pytest.raises(AttributeError):
        del prim.children


## Hierarchy invariants
@pytest.mark.parametrize(
    "prim,hierarchy_depth",
    [
        (CompositePrimitive(), 1),
        (CompositePrimitive(), 3),
        (SimplePrimitive(), 1),
        (SimplePrimitive(), 3),
        (AtomicPrimitive(ELEMENTS[5]), 1),
        (AtomicPrimitive(ELEMENTS[5]), 3),
    ],
)
def test_frozen_hierarchy_propagates(prim: Primitive, hierarchy_depth: int) -> None:
    """Test that hierarchy modification state changes bubble up through hierarchy"""
    root = dummy_hierarchy_atop_prim(prim, num_intermed=hierarchy_depth)
    # N.B.: deliberately NOT freezing at prim; should propagate down
    root.freeze_hierarchy()

    with pytest.raises(AttributeError):
        prim.parent = root


@pytest.mark.parametrize(
    "parent,child",
    [
        (SimplePrimitive(), RootPrimitive()),
        (CompositePrimitive(), RootPrimitive()),
        (SimplePrimitive(), CompositePrimitive()),
    ],
)
def test_improper_hierarchy_parent_disallowed(
    parent: Primitive,
    child: Primitive,
) -> None:
    """
    Test that assigning illegal parent-child relationships
    between Primitives through the 'parent' field is disallowed
    """
    with pytest.raises((ArborescenceError, ImproperHierarchyError)):
        child.parent = parent


@pytest.mark.parametrize(
    "parent,child",
    [
        (SimplePrimitive(), RootPrimitive()),
        (CompositePrimitive(), RootPrimitive()),
        (SimplePrimitive(), CompositePrimitive()),
    ],
)
def test_improper_hierarchy_children_disallowed(
    parent: Primitive,
    child: Primitive,
) -> None:
    """
    Test that assigning illegal parent-child relationships
    between Primitives through the 'children' field is disallowed
    """
    with pytest.raises((ArborescenceError, ImproperHierarchyError)):
        parent.children = [child]


# Hierarchy assembly
def test_hierarchy_assembly():
    """
    Test that Primitives can be assembled into a hierarchy,
    and that the resulting hierarchy looks as anticipated
    """
    root = RootPrimitive()
    comp_0 = CompositePrimitive()
    comp_1 = CompositePrimitive()
    simple_0 = SimplePrimitive()
    simple_1 = SimplePrimitive()
    simple_2 = SimplePrimitive()
    simple_3 = SimplePrimitive()

    simple_0.parent = root
    comp_0.parent = root
    simple_1.parent = comp_0
    comp_1.parent = comp_0
    comp_1.children = [simple_2, simple_3]  # also test that this mechanism works

    ancestry_expected: dict[SimplePrimitive, tuple[SupportsChildren, ...]] = {
        simple_0: (root,),
        simple_1: (root, comp_0),
        simple_2: (root, comp_0, comp_1),
        simple_3: (root, comp_0, comp_1),
    }

    for simple, ancestors_expected in ancestry_expected.items():
        assert simple.ancestors == ancestors_expected


# hierarchy_tree tests: origniated from this discussion:
# https://github.com/MuPT-hub/mupt/pull/135#discussion_r4393946041
def test_hierarchy_tree_includes_constructor_children() -> None:
    """Keep constructor-supplied children in the exported hierarchy"""
    child = CompositePrimitive(label="child")
    root = RootPrimitive(label="root", children=[child])

    graph = root.hierarchy_tree()

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge(root, child)


def test_hierarchy_tree_preserves_equivalent_siblings() -> None:
    """Keep separate hierarchy nodes for equivalent sibling Primitives"""
    root = RootPrimitive(label="root")
    left = SimplePrimitive(label="left")
    right = SimplePrimitive(label="right")
    root.attach_child(left)
    root.attach_child(right)

    graph = root.hierarchy_tree()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
