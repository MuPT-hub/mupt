"""Unit tests for Connectors, AttachmentPoints, and connection-related utilities"""

import pytest

from mupt.chemistry.core import BondType
from mupt.mupr.connection import Connector, AttachmentPoint
from mupt.builders.heading import TraversalDirection


# bondability tests
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
        (C1, C4, True),   # should NOT fail, compatible connectors
    ],
)
def test_connector_bondability(
    conn1: Connector, conn2: Connector, expected_bondable: bool
) -> None:
    """Test bondability checks between two Connectors"""
    assert Connector.bondable_with(conn1, conn2) == expected_bondable


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
    ],
)
def test_connector_counterpart_bondable(conn: Connector) -> None:
    """
    Test that the co-Connector produced by Connector.counterpart()
    is bondable to the original when attachment points are nonempty
    """
    conn_empty = (not conn.anchor.attachables) or (
        not conn.linker.attachables
    )  # False only when nonempty
    counterpart_bondable = Connector.bondable_with(conn, conn.counterpart())

    # XOR, since conditions are mutually-exclusive
    assert (conn_empty ^ counterpart_bondable)  
