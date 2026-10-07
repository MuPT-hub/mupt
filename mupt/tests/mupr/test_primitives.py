"""Unit tests for Primitive interactions with one another and with sub-components"""
# ruff: noqa: D103 (missing docstrings on tests is OK)

import pytest

from itertools import product as cartesian
import numpy as np

from mupt.mutils.iteration import sliding_window
from mupt.chemistry.core import ELEMENTS
from mupt.trees.subselect import NodePredicate

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
    SupportsParents,
    RootPrimitive,
    CompositePrimitive,
    SimplePrimitive,
    AtomicPrimitive,
)


# Helpers
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


def hierarchy_example() -> RootPrimitive:
    """
    A moderately-complex but still-small hierarchy which contains
    all Primitive subtypes and varied parent/child relationships
    """
    # Connectors
    conn0 = Connector(
        anchor=AttachmentPoint({1}),
        linker=AttachmentPoint({2}),
        bondtype=BondType.DOUBLE,
    )
    conn1 = Connector(
        anchor=AttachmentPoint({1}),
        linker=AttachmentPoint({2}),
        bondtype=BondType.AROMATIC,
    )

    # Primitive parts
    root = RootPrimitive()
    comp_0 = CompositePrimitive()
    comp_1 = CompositePrimitive()
    comp_2 = CompositePrimitive()
    simple_0 = SimplePrimitive(connections=(conn0.copy(),))
    simple_1 = SimplePrimitive(
        connections=(
            conn0.counterpart(),
            conn0.copy(),
            conn1.copy(),
            conn0.counterpart(),
        )
    )
    simple_2 = SimplePrimitive(
        connections=(
            conn0.counterpart(),
            conn0.copy(),
            conn1.copy(),
        )
    )
    atom_1 = AtomicPrimitive(
        element=ELEMENTS[6],
        connections=(
            conn1.counterpart(),
            conn1.counterpart(),
        ),
    )

    # Assembly
    simple_0.parent = root
    comp_0.parent = root
    comp_1.parent = comp_0
    simple_1.parent = comp_1
    comp_2.parent = root
    comp_2.children = [simple_2, atom_1]  # also test that this mechanism works
    print(root.hierarchy_summary())

    # Connection
    simple_0.connect_neighbor(simple_1)
    simple_1.connect_neighbor(simple_2)
    simple_1.connect_neighbor(simple_2)
    simple_1.connect_neighbor(atom_1)
    simple_2.connect_neighbor(atom_1)

    return root


# Data security
@pytest.mark.parametrize(
    "prim",
    [
        RootPrimitive(),
        CompositePrimitive(),
        SimplePrimitive(),
        AtomicPrimitive(ELEMENTS[2]),
    ],
)
def test_frozen_add_connectors(prim: Primitive) -> None:
    """Test that connector addition is blocked by freezing it on any Primitive"""
    prim.freeze_connections()
    with pytest.raises(AttributeError):
        prim._add_connector(Connector())


@pytest.mark.parametrize(
    "prim",
    [
        RootPrimitive(),
        CompositePrimitive(),
        SimplePrimitive(),
        AtomicPrimitive(ELEMENTS[2]),
    ],
)
def test_frozen_remove_connectors(prim: Primitive) -> None:
    """Test that connector removal is blocked by freezing it on any Primitive"""
    prim.unfreeze_connections()
    conn_addr = prim._add_connector(Connector())
    prim.freeze_connections()
    with pytest.raises(AttributeError):
        prim._remove_connector(conn_addr)


def test_frozen_connectors_propagates() -> None:
    """Test that connector modification state changes bubble up through hierarchy"""
    root = RootPrimitive()
    comp = CompositePrimitive()
    simple = SimplePrimitive()

    comp.parent = root
    simple.parent = comp

    root.freeze_connections()
    with pytest.raises(AttributeError):
        simple._add_connector(Connector())


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


# Combining Primitives into hierarchy
@pytest.mark.parametrize(
    "prim,expected_simple,expected_supports_children,expected_supports_parents",
    [
        (RootPrimitive(), False, True, False),
        (CompositePrimitive(), False, True, True),
        (SimplePrimitive(), True, False, True),
        (AtomicPrimitive(ELEMENTS[1]), True, False, True),
    ],
)
def test_hierarchy_declarations(
    prim: Primitive,
    expected_simple: bool,
    expected_supports_children: bool,
    expected_supports_parents: bool,
) -> None:
    """
    Test whether instances of particular types of Primitive correctly
    declare their capacity in a representation hierarchy
    """
    assert (
        (prim.is_simple == expected_simple)
        and (prim.supports_children == expected_supports_children)
        and (prim.supports_parents == expected_supports_parents)
    )


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


# Inserting and withdrawing Connectors from a hierarchy
@pytest.mark.parametrize(
    "simple,num_intermed",
    [
        # direct-to-root (no intermediates)
        (SimplePrimitive(), 0),
        ## test that other Connector instances aren't a distraction
        (SimplePrimitive(connections=[Connector()]), 0),
        (AtomicPrimitive(element=ELEMENTS[1]), 0),
        # 3 intermediate Composite levels, to imitate more complex hierarchy
        (SimplePrimitive(), 3),
        (SimplePrimitive(connections=[Connector()]), 3),
        (AtomicPrimitive(element=ELEMENTS[1]), 3),
    ],
)
def test_simple_add_connector(
    simple: SimplePrimitive,
    num_intermed: int,
) -> None:
    """
    Test that adding Connectors to any kind of SimplePrimitive:
    b) sets that Connectors .holder attribute to itself
    a) adds that Connector to the Simples managed pool of Connectors
    c) injects that Connector through any parent levels of a hierarchy
    """
    _root = dummy_hierarchy_atop_prim(simple, num_intermed=num_intermed)
    connector = Connector()
    simple.add_connector(connector)

    assert connector.holder == simple
    for prim in simple.path:
        assert connector in prim.connections.connectors


@pytest.mark.parametrize(
    "simple,num_intermed",
    [
        # direct-to-root (no intermediates)
        (SimplePrimitive(), 0),
        ## test that other Connector instances aren't a distraction
        (SimplePrimitive(connections=[Connector()]), 0),
        (AtomicPrimitive(element=ELEMENTS[1]), 0),
        # 3 intermediate Composite levels, to imitate more complex hierarchy
        (SimplePrimitive(), 3),
        (SimplePrimitive(connections=[Connector()]), 3),
        (AtomicPrimitive(element=ELEMENTS[1]), 3),
    ],
)
def test_simple_remove_connector(
    simple: SimplePrimitive,
    num_intermed: int,
) -> None:
    """
    Test that removing Connectors from any kind of SimplePrimitive:
    b) unsets the Connectors .holder attribute
    a) removes that Connector from the Simples managed pool of Connectors
    c) withdraws that Connector from all parent levels of a hierarchy
    """
    _root = dummy_hierarchy_atop_prim(simple, num_intermed=num_intermed)
    connector = Connector()
    simple.add_connector(connector)  # TB: add_connector should have been tested prior

    simple.remove_connector(connector)

    assert connector.holder is None
    for prim in simple.path:
        assert connector not in prim.connections.connectors


def test_simple_remove_connector_nonexistent() -> None:
    """
    Test that attempting to remove a Connector which was
    never there to begin with is caught and raises Exception
    """
    simple = SimplePrimitive()
    connector = Connector()

    with pytest.raises(KeyError):
        simple.remove_connector(connector)


# Setting neighbors and topologies
def test_connect_neighbor() -> None:
    """Test that new connections to a neighbor Primitive can be made"""
    simple = SimplePrimitive()
    comp = CompositePrimitive(children=[simple])
    root = RootPrimitive(children=[comp])

    simple_new = SimplePrimitive()
    simple_new.parent = root

    conn = basic_connector()
    conn_counter = conn.counterpart()
    simple.add_connector(conn)
    simple_new.add_connector(conn_counter)

    # 1) check no neighbors possible before making connection
    assert not any(simple_new.potential_neighbors())

    simple_new.connect_neighbor(
        simple,
        our_connector=conn_counter,
        their_connector=conn,
    )

    # 2) check that ALL Primitives on parallel branch are potential neighbors
    assert set(*simple_new.potential_neighbors()) == set([comp, simple])


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
    """
    Test that neighborship status automatically multiscales

    I.e. given two distinct branches of the hierrachy tree,
    any pair of Primitives, one from either branch" being assigned neighbors
    automatically makes EVERY pair from those branches neighors as well
    """
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


# Sub-selecting Primitives
def test_primitive_predicates():
    """
    Test that subselecting descendants of a Primitive
    by a predicate returns the expected results and
    preserves the 'one-selection-per-branch' invariant
    """
    ...


def test_potential_neighbors():
    """
    Test that all non-internal neighbors of a
    Primitive are correctly identified in a hierarchy
    """
    ...


def test_neighbors():
    """Test resolution-specific (i.e. predicate-based) neighbor selection"""
    ...


# TB: this ought to have many tests
@pytest.mark.parametrize(
    "root,predicate",
    [],
)
def test_cross_section_nodes(
    root: SupportsChildren,
    predicate: NodePredicate[Primitive],
) -> None:
    """
    Test that the Primitives primoprogenitors
    selected for by a chosen predicate are made
    the nodes of the cross-section graph
    """
    ...


@pytest.mark.parametrize(
    "root,predicate",
    [],
)
def test_cross_section_edges(
    root: SupportsChildren,
    predicate: NodePredicate[Primitive],
) -> None:
    """
    Test that neighboring Primitives in a cross
    section graph are each spanned by an edge
    """
    ...


def test_cross_section_bond_orders():
    """
    Test that bond order info is completely transferred
    to the edges of the cross-section graph
    """
    ...


## Copying
def test_primitive_copy_hierarchy(hierarchy_example: RootPrimitive) -> None:
    """Test that hierarchy of copy is isomorphic to that of the original"""
    # copy = example_hierarchy.copy()
    ...


def test_primitive_copy_connectors(hierarchy_example: RootPrimitive) -> None:
    """Test that Connectors on copy are analogous to original WITHOUT being identical"""
    clone = hierarchy_example.copy()
    for conn_orig, conn_copy in zip(
        hierarchy_example.connectors,
        clone.connectors,
    ):
        # TB: slightly brittle, since assumes copied connectors
        # will bein same order as in the original; revisit if that
        # assumption becomes invalid (this test will fail if so)
        assert (
            Connector.fungible_with(conn_copy, conn_orig)
            and (conn_copy is not conn_orig)  # equilvane,t but not identical
        )


# System info on Roots
def test_root_default_box_vectors() -> None:
    """Test that root can store box vector array and that it has a sensible default"""
    root = RootPrimitive()

    assert np.allclose(root.box_vectors, np.eye(3, dtype=float))


# Resolution shifts on (mutable) Composites
...
