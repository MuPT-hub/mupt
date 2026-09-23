"""Unit tests for Primitive interactions with one another and with sub-components"""
# ruff: noqa: D103 (missing docstrings on tests is OK)

import pytest

from itertools import product as cartesian
import numpy as np

from mupt.mupr.connection.connectors import (
    Connector,
    AttachmentPoint,
    BondType,
)
from mupt.mupr.primitives import (
    ArborescenceError,
    ImproperHierarchyError,
    Primitive,
    SupportsChildren,
    RootPrimitive,
    CompositePrimitive,
    SimplePrimitive,
)


def basic_connector() -> Connector:
    """
    A simple Connector schema to use in tests
    where the precise Connector isn't important
    """
    # DEV: Connectors obtained from consecutive calls will
    # be fungible, but not identical (different instances)
    return Connector(
        anchor=AttachmentPoint({1}),
        linker=AttachmentPoint({2}),
        bondtype=BondType.DOUBLE,
    )


# Combining Primitives into hierarchy
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


@pytest.mark.parametrize(
    "parent,child",
    [
        (SimplePrimitive(), RootPrimitive()),
        (CompositePrimitive(), RootPrimitive()),
        (SimplePrimitive(), CompositePrimitive()),
    ],
)
def test_improper_hierarchy_disallowed(parent: Primitive, child: Primitive) -> None:
    """Test that illegal parent-child relationships among Primitives are disallowed"""
    with pytest.raises((ArborescenceError, ImproperHierarchyError)):
        child.parent = parent


def test_frozen_hierarchy():
    """Test that hierarchy modification is blocked by freezing it on any Primitive"""
    ...


# Setting neighbors and topologies
@pytest.mark.parametrize(
    "",
    [],
)
def test_connect_neighbor(): ...


def test_positive_is_neighbors_with_symmetric():
    """
    Test neighborship check is indeed invariant under swapping Primitive arguments
    in the case that the two Primitives involved ARE neighbors (both positive)
    """
    prim_0 = SimplePrimitive()
    conn = basic_connector()
    prim_0.add_connector(conn)

    prim_1 = SimplePrimitive()
    conn_counter = conn.counterpart()
    prim_1.add_connector(conn_counter)

    prim_0.connect_neighbor(
        prim_1,
        # TB: no need to specify connectors; linker only has one choice
        our_connector=conn,
        their_connector=conn_counter,
    )

    assert prim_0.is_neighbors_with(prim_1) and prim_1.is_neighbors_with(prim_0)


def test_negative_is_neighbors_with_symmetric():
    """
    Test neighborship check is indeed invariant under swapping Primitive arguments
    in the case that the two Primitives involved ARE NOT neighbors (both negative)
    """
    prim_0 = CompositePrimitive()
    prim_1 = SimplePrimitive()

    assert not (prim_0.is_neighbors_with(prim_1) or prim_1.is_neighbors_with(prim_0))


def test_neighborship_propagates_thru_hierarchy():
    root_0 = RootPrimitive()
    comp_0 = CompositePrimitive()
    simp_0 = SimplePrimitive()
    comp_0.parent = root_0
    simp_0.parent = comp_0
    conn_0 = basic_connector()
    simp_0.add_connector(conn_0)  # should also be registered up thru hierarchy

    root_1 = RootPrimitive()
    comp_1 = CompositePrimitive()
    simp_1 = SimplePrimitive()
    comp_1.parent = root_1
    simp_1.parent = comp_1
    conn_1 = conn_0.counterpart()
    simp_1.add_connector(conn_1)  # should also be registered up thru hierarchy

    # connect between hierarchy levels for the hat trick :P
    comp_0.connect_neighbor(simp_1)

    # by design, if any pair of Primitives from the two parallel branches are
    # neighbors, then so is EVERY possible pair of Primitives between those branches
    for prim_0, prim_1 in cartesian(simp_0.path, simp_1.path):
        # opting against call w/ prim_0 as 'self' to emphasize symmetry of args
        assert Primitive.is_neighbors_with(prim_0, prim_1)


def test_frozen_connectors(): ...


# Sub-selecting Primitives
def test_primitive_predicates(): ...


def test_neighbors_unconditional(): ...


def test_neighbors_subset(): ...


def test_cross_section(): ...


# System info on Roots
def test_root_default_box_vectors() -> None:
    """Test that root can store box vector array and that it has a sensible default"""
    root = RootPrimitive()

    assert np.allclose(root.box_vectors, np.eye(3, dtype=float))


# Resolution shifts on (mutable) Composites
...


# Inserting Connectors into and deleting Connectors from hierarchy on Simples
def test_simple_add_connector(): ...


def test_simple_remove_connector(): ...


def test_simple_inject_connector_into_hierarchy(): ...


def test_simple_withdraw_connector_from_hierarchy(): ...
