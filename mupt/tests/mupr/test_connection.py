"""Unit tests for Connectors, AttachmentPoints, and connection-related utilities"""

import pytest

from networkx.algorithms import equivalence_classes

from mupt.chemistry.core import BondType
from mupt.builders.heading import TraversalDirection
from mupt.mupr.connection.exceptions import ConnectorLockedError
from mupt.mupr.connection import Connector, AttachmentPoint


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
        counterpart.linker.attachables == connector.anchor.attachables
        and counterpart.anchor.attachables == connector.linker.attachables
        and counterpart.bondtype == connector.bondtype
    )
    # TODO: compare positional info?
    # and counterpart.tangent_vector == connector.tangent_vector


@pytest.mark.parametrize(
    "conn",
    [
        # test with the empty connector the verify that counterpart
        # bondability fails when attachment points are empty
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
    is bondable to the original when attachment points are nonempty
    """
    # False only when nonempty
    conn_empty = (not conn.anchor.attachables) or (not conn.linker.attachables)
    counterpart_bondable = Connector.bondable_with(conn, conn.counterpart())

    # XOR, since conditions are mutually-exclusive
    assert conn_empty ^ counterpart_bondable


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
        ),  # should fail, empty two attachment point sets must be disjoint
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


def test_connector_lock_blocks_delete() -> None:
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
