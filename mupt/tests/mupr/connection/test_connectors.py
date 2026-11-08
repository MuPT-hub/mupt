"""Unit tests for Connector objects"""

import pytest
from numpy.testing import assert_allclose

from typing import Optional

from networkx.algorithms import equivalence_classes

from . import randomly_placed_connector
from mupt.chemistry.core import BondType
from mupt.builders.heading import TraversalDirection
from mupt.geometry.coordinates.directions import random_tangent
from mupt.mupr.connection.connectors import Connector, AttachmentPoint
from mupt.mupr.connection.exceptions import ConnectorLockedError
from mupt.mupr.connection.management import (
    ConnectorManager,
    ConnectorManagerFrozen,
)


# Counterpart tests
CA = Connector(
    anchor=AttachmentPoint({1}),
    linker=AttachmentPoint({2}),
    bondtype=BondType.DOUBLE,
)
# variant of CA with positional info; should compare as non-coincident
CB = Connector(
    anchor=AttachmentPoint({1}),
    linker=AttachmentPoint({2}, position=[1, 2, 3]),
    bondtype=BondType.DOUBLE,
)
CC = Connector(
    anchor=AttachmentPoint({2}),
    linker=AttachmentPoint({1, 3}),
    bondtype=BondType.AROMATIC,
)
CD = Connector(
    anchor=AttachmentPoint({"same"}),
    linker=AttachmentPoint({"same"}),
    bondtype=BondType.SINGLE,
)


@pytest.mark.parametrize(
    "connector",
    [CA, CB, CC, CD],
)
def test_counterpart(connector: Connector):
    """Test that a Connector's counterpart has the expected anchor/linker attributes"""
    counterpart = connector.counterpart()

    assert (
        counterpart.linker.markers == connector.anchor.markers
        and counterpart.anchor.markers == connector.linker.markers
        and counterpart.bondtype == connector.bondtype
    )
    # TODO: compare positional info?
    # and counterpart.tangent_vector == connector.tangent_vector


def test_counterpart_non_identical() -> None:
    """
    Test that anchor and linker of counterpart are equivalent to
    the linker and anchor, respectively, of the original Connector
    WITHOUT being literally the same instances
    """
    conn_original = Connector(
        anchor=AttachmentPoint({1}, position=[1.0, 2.0, 3.0]),
        linker=AttachmentPoint({2}, position=[4.0, 5.0, 6.0]),
        bondtype=BondType.DOUBLE,
    )
    conn_counter = conn_original.counterpart()

    for att_point_original, att_point_counter in (
        (conn_original.anchor, conn_counter.linker),
        (conn_original.linker, conn_counter.anchor),
    ):
        assert att_point_original == att_point_counter
        assert att_point_original is not att_point_counter


@pytest.mark.parametrize(
    "conn",
    [
        # test with the empty connector the verify that counterpart
        # bondability fails when AttachmentPoint ends are empty
        Connector(),
        Connector(
            anchor=AttachmentPoint({"a", "b", TraversalDirection.RETRO}),
            linker=AttachmentPoint({"c", TraversalDirection.ANTERO}),
            bondtype=BondType.SINGLE,
        ),
        CA,
        CB,
        CC,
        CD,
    ],
)
def test_connector_counterpart_bondable(conn: Connector) -> None:
    """
    Test that the co-Connector produced by Connector.counterpart()
    is bondable to the original when AttachmentPoint objects are nonempty
    """
    # False only when nonempty
    conn_empty = (not conn.anchor.markers) or (not conn.linker.markers)
    counterpart_bondable = Connector.bondable_with(conn, conn.counterpart())

    # XOR, since conditions are mutually-exclusive
    assert conn_empty ^ counterpart_bondable


def test_counterpart_orthogonal_basis() -> None:
    """Ensure that a Connector's counterpart is geometrically compatible with itself"""
    conn = randomly_placed_connector()
    conn.tangent_vector = random_tangent(conn.bond_vector)
    conn_counter = conn.counterpart()

    assert_allclose(conn.bond_vector / conn_counter.bond_vector, -1.0)
    assert_allclose(conn.tangent_vector / conn_counter.tangent_vector, -1.0)
    assert_allclose(conn.normal_vector / conn_counter.normal_vector, 1.0)


def test_individualize() -> None:
    """Test that Connector.individualize gives the expected set of derived Connectors"""
    conn_master = Connector(AttachmentPoint({0, 1, 2}), AttachmentPoint({3, 4}))
    conns_expected: dict[tuple[int, int], Connector] = {
        (0, 3): Connector(AttachmentPoint({0}), AttachmentPoint({3})),
        (1, 3): Connector(AttachmentPoint({1}), AttachmentPoint({3})),
        (2, 3): Connector(AttachmentPoint({2}), AttachmentPoint({3})),
        (0, 4): Connector(AttachmentPoint({0}), AttachmentPoint({4})),
        (1, 4): Connector(AttachmentPoint({1}), AttachmentPoint({4})),
        (2, 4): Connector(AttachmentPoint({2}), AttachmentPoint({4})),
    }

    indivs = conn_master.individualize()
    assert indivs.keys() == conns_expected.keys()
    for key, conn_expected in conns_expected.items():
        assert Connector.fungible_with(indivs[key], conn_expected)


# Comparison tests
## only use counterparts in subsequent tests if prior counterpart tests have passed
## DEV: creating here since expected result must contain same literal Connector instance
CA_COPY = CA.copy()
CA_COUNTERPART = CA.counterpart()

CB_COPY = CB.copy()
CB_COUNTERPART = CB.counterpart()

CC_COPY = CC.copy()
CC_COUNTERPART = CC.counterpart()

CD_COPY = CD.copy()
CD_COUNTERPART = CD.counterpart()


@pytest.mark.parametrize(
    "connectors,equiv_classes_expected",
    [
        (  # all 3 example prototype are non-fungible
            (CA, CB, CC, CD),
            {
                frozenset([CA]),
                frozenset([CB]),
                frozenset([CC]),
                frozenset([CD]),
            },
        ),
        (  # test that copies are practially indistinguishable from originals
            (
                CA,
                CA_COPY,
                CB,
                CB_COPY,
                CC,
                CC_COPY,
            ),
            {
                frozenset([CA, CA_COPY]),
                frozenset([CB, CB_COPY]),
                frozenset([CC, CC_COPY]),
            },
        ),
        (  # these Connectors are distinct from their counterparts
            (CA, CA_COUNTERPART, CB, CB_COUNTERPART),
            {
                frozenset([CA]),
                frozenset([CA_COUNTERPART]),
                frozenset([CB]),
                frozenset([CB_COUNTERPART]),
            },
        ),
        (  # this Connector is actually designed to be equivalent to its counterpart
            (CD, CD_COUNTERPART),
            {frozenset([CD, CD_COUNTERPART])},
        ),
    ],
)
def test_connector_fungibility(
    connectors: tuple[Connector, ...],
    equiv_classes_expected: set[frozenset[Connector]],
) -> None:
    """
    Test that total comparison of Connectors for fungibility
    (i.e. interchangeability) groups together Connectors as expected
    """
    equiv_classes_actual = equivalence_classes(
        connectors,
        relation=Connector.fungible_with,
    )

    assert equiv_classes_actual == equiv_classes_expected


C1 = Connector(
    anchor=AttachmentPoint({"a"}),
    linker=AttachmentPoint({"z"}),
    bondtype=BondType.DOUBLE,
)
C2 = Connector(
    anchor=AttachmentPoint({"b"}),
    linker=AttachmentPoint({"a"}),
    bondtype=BondType.SINGLE,
)
C3 = Connector(
    anchor=AttachmentPoint({"z"}),
    linker=AttachmentPoint({"a"}),
    bondtype=BondType.SINGLE,
)
C4 = Connector(
    anchor=AttachmentPoint({"z"}),
    linker=AttachmentPoint({"a"}),
    bondtype=BondType.DOUBLE,
)


@pytest.mark.parametrize(
    "conn1, conn2, expected_bondable",
    [
        (
            Connector(),
            Connector(),
            False,
        ),  # should fail, two empty AttachmentPoint marker sets must be disjoint
        (C1, C2, False),  # should fail, anchorables of C1 not in linkables of C1
        (C1, C3, False),  # should fail, bond types differ
        (C1, C4, True),  # should NOT fail, compatible connectors
    ],
)
def test_connector_bondability(
    conn1: Connector, conn2: Connector, expected_bondable: bool
) -> None:
    """Test bondability checks between two Connectors"""
    assert Connector.bondable_with(conn1, conn2) == expected_bondable


# Neighbor tests
def test_connector_lock() -> None:
    """Test that locking a Connector actually locks it"""
    connector = Connector()
    connector.lock()

    assert connector.is_locked


def test_connector_unlock() -> None:
    """Test that unlocking a Connector actually unlocks it"""
    connector = Connector()
    connector.unlock()

    assert not connector.is_locked


def test_connector_toggle_lock() -> None:
    """Test that toggling a Connectors lock actually inverts it lock status"""
    connector = Connector()
    lock_status_init = connector.is_locked
    connector.toggle_lock()

    assert connector.is_locked is not lock_status_init


def test_connector_lock_blocks_write() -> None:
    """Test that a locked connector cannot be written to"""
    # DEV: deliberately not fixture, though reused; want to quarantine attr modification
    conn0 = Connector(
        anchor=AttachmentPoint({1, 2}),
        linker=AttachmentPoint({3, 4}),
    )
    conn0.lock()

    conn1 = conn0.counterpart()

    with pytest.raises(ConnectorLockedError):
        conn0.neighbor = conn1


def test_neighbor_assignment():
    """Test that (unlocked) Connectors can be mutually assigned neighbors each other"""
    conn0 = Connector(
        anchor=AttachmentPoint({1, 2}),
        linker=AttachmentPoint({3, 4}),
    )
    conn0.unlock()  # double-check both are unlocked

    conn1 = conn0.counterpart()
    conn1.unlock()  # double-check both are unlocked

    conn0.neighbor = conn1

    assert (conn0.neighbor == conn1) and (conn1.neighbor == conn0)


def test_connector_neighbor_persistence() -> None:
    """
    Test that the former neighbor of a Connector 'forgets'
    the aforementioned Connector when its neighbor is rebound
    """
    conn_a = Connector()
    conn_b = Connector()
    conn_c = Connector()

    conn_a.neighbor = conn_b
    conn_a.neighbor = conn_c

    assert conn_b.neighbor is not conn_a
    assert conn_b.neighbor is None


def test_connector_lock_blocks_neighbor_delete() -> None:
    """Test that a locked connector cannot have an existing neighbor removed"""
    conn0 = Connector(
        anchor=AttachmentPoint({1, 2}),
        linker=AttachmentPoint({3, 4}),
    )
    conn0.unlock()

    conn1 = conn0.counterpart()
    conn0.neighbor = conn1
    conn1.lock()

    with pytest.raises(ConnectorLockedError):
        del conn0.neighbor


# Holders
## Helpers
class MockHolder:
    """Dummy holder for testing"""

    connections: ConnectorManager

    def __init__(self, label: str, connector_manager: ConnectorManager) -> None:
        self.label = label
        self.connector_manager = connector_manager


def mock_holder() -> MockHolder:
    """Holder example"""
    return MockHolder(label="fixture", connector_manager=ConnectorManagerFrozen())


def conn_without_holder() -> Connector:
    """Connector with NO holder assigned"""
    return Connector()


def conn_without_holder_locked() -> Connector:
    """Connector with NO holder assigned, and modification locked"""
    conn = conn_without_holder()
    conn.lock()

    return conn


def conn_with_holder() -> Connector:
    """Connector with holder assigned"""
    conn = conn_without_holder()
    conn.holder = mock_holder()

    return conn


def conn_with_holder_locked() -> Connector:
    """Connector with holder assigned, and modification locked"""
    conn = conn_with_holder()
    conn.lock()

    return conn


## Tests
@pytest.mark.parametrize("new_holder", [None, mock_holder()])
def test_holder_set_unlocked(new_holder: Optional[MockHolder]) -> None:
    """Test holder assignment"""
    conn = conn_without_holder()
    conn.holder = new_holder
    assert conn._holder == new_holder


@pytest.mark.parametrize(
    "conn", [conn_with_holder_locked(), conn_without_holder_locked()]
)
def test_holder_set_locked(conn: Connector) -> None:
    """Test that locked Connectors block one from editing their holder"""
    holder = MockHolder(label="local", connector_manager=ConnectorManagerFrozen())
    with pytest.raises(ConnectorLockedError, match="holder.*locked Connector"):
        conn.holder = holder


@pytest.mark.parametrize(
    "conn,has_holder_expected",
    [
        (conn_without_holder(), False),
        (conn_without_holder_locked(), False),
        (conn_with_holder(), True),
        (conn_with_holder_locked(), True),
    ],
)
def test_has_holder(conn: Connector, has_holder_expected: bool) -> None:
    """Test holder check works"""
    assert conn.has_holder is has_holder_expected
