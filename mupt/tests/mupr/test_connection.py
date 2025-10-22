'''Unit tests for Connectors, AttachmentPoints, and connection-related utilities'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest

from mupt.chemistry.core import BondType
from mupt.mupr.connection import Connector, AttachmentPoint, AttachmentLabel, TraversalDirection


@pytest.mark.parametrize(
    'conn1, conn2, expected_bondable', 
    [ # TODO: populate with examples
        # (Connector(), Connector(), True)
    ]
)
def test_connector_bondability(conn1 : Connector, conn2 : Connector, expected_bondable : bool) -> None:
    '''Test bondability checks between two Connectors'''
    assert Connector.bondable_with(conn1, conn2) == expected_bondable

@pytest.mark.parametrize(
    'conn', 
    [
        Connector(), # test with the empty connector the verify that counterpart bondability fails when attachment points are empty
        Connector(
            anchor=AttachmentPoint({'a', 'b', TraversalDirection.RETRO}),
            linker=AttachmentPoint({'c', TraversalDirection.ANTERO}),
            bondtype=BondType.SINGLE,
        ),
    ]
)
def test_connector_counterpart_bondable(conn : Connector) -> None:
    '''
    Test that the co-Connector produced by Connector.counterpart() 
    is bondable to the original when attachment points are nonempty
    '''
    conn_empty = (not conn.anchor.attachables) or (not conn.linker.attachables) # False only when nonempty
    counterpart_bondable = Connector.bondable_with(conn, conn.counterpart())
    
    assert (conn_empty ^ counterpart_bondable) # XOR, since conditions are mutually-exclusive
    