"""Unit tests for multiscale-aware neighbor connections between on MuPT Primitives"""

import pytest

from itertools import product as cartesian

from mupt.chemistry.core import ELEMENTS, BondType
from mupt.mutils.iteration import sliding_window

from mupt.mupr.primitives.primitive import Primitive
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.simple.atomic import AtomicPrimitive
from mupt.mupr.connection.connectors import Connector, AttachmentPoint


# Fixtures
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


# Tests proper
## Data security
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


# Neighbor relations
def test_potential_neighbors():
    """
    Test that all non-internal neighbors of a
    Primitive are correctly identified in a hierarchy
    """
    ...


def test_neighbors():
    """Test resolution-specific (i.e. predicate-based) neighbor selection"""
    ...


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
    comp_0.connect_neighbor(simp_1, our_connector=conn_0, their_connector=conn_1)

    # by design, if any pair of Primitives from the two parallel branches are
    # neighbors, then so is EVERY possible pair of Primitives between those branches
    for prim_0, prim_1 in cartesian(simp_0.path, simp_1.path):
        # opting against call w/ prim_0 as 'self' to emphasize symmetry of args
        assert Primitive.is_neighbors_with(prim_0, prim_1)


## Connector modification
def test_simple_remove_connector_nonexistent() -> None:
    """
    Test that attempting to remove a Connector which was
    never there to begin with is caught and raises Exception
    """
    simple = SimplePrimitive()
    connector = Connector()

    with pytest.raises(KeyError):
        simple.remove_connector(connector)


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


def test_moving_linked_primitive_keeps_neighbor_antialigned() -> None:
    """
    Moving a Primitive whose Connector is already linked re-aims that Connector
    at its (fixed) neighbor instead of raising (one-sided: the neighbor is not moved)
    """
    import numpy as np
    from scipy.spatial.transform import RigidTransform

    from mupt.interfaces.smiles import primitive_from_smiles
    from mupt.mupr.connection.alignment import ConnectorAntialignmentRigid

    head = primitive_from_smiles("[H]-[CH2]-[CH2:1]-[*:2]", embed_positions=True)
    tail = primitive_from_smiles("[*:1]-[CH2:2][CH3]", embed_positions=True)
    chain = CompositePrimitive(children=[head, tail])

    ours, theirs = next(
        (x, y)
        for x in head.connections.connectors_free
        for y in tail.connections.connectors_free
        if x.bondable_with(y)
    )
    ours.set_bond_length(1.54)
    theirs.set_bond_length(1.54)
    tail.rigidly_transform(
        ConnectorAntialignmentRigid().antialignment_transformation(
            align_connector=theirs, to_connector=ours
        )
    )
    head.connect_neighbor(tail, our_connector=ours, their_connector=theirs)

    tail.rigidly_transform(RigidTransform.from_translation(np.array([0.5, 0.0, 0.0])))
    assert ours.neighbor is theirs
    assert chain is head.parent
    # the moved Connector's linker is re-aimed onto its fixed neighbor's anchor
    assert np.allclose(theirs.linker.position, ours.anchor.position, atol=1e-6)
