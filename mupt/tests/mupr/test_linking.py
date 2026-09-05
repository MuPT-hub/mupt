'''Unit tests for Connector to graph edge linking protocol(s)'''

import pytest
from dataclasses import dataclass

from networkx import Graph, cycle_graph

from mupt.mupr.linking import (
    deduce_connections_from_topology,
    EdgeMissingError,
)
from mupt.chemistry.core import BondType
from mupt.mupr.connection.connectors import Connector, AttachmentPoint


@dataclass
class LinkerTestExample:
    '''Encapsulation for input data to linker protocol'''
    topology : Graph # Graph[int]
    connector_assignments : dict[int, tuple[Connector, ...]]
    possible_solutions : tuple[
        dict[
            tuple[int, int],
            tuple[Connector, Connector]
        ],
        ...
    ]
    
def triangle_example() -> LinkerTestExample:
    '''
    3-node cyclic graph example which tests that unique, non-greedy match info propagates to
    subsequent match tests, and that distinct-but-equivalent Connectors are treated as a unit
    '''
    # reference Connectors
    conn_type_a = Connector(
        anchor=AttachmentPoint(attachables={1}),
        linker=AttachmentPoint(attachables={2}),
        bondtype=BondType.SINGLE,
    )
    conn_type_b = Connector(
        anchor=AttachmentPoint(attachables={2, 3}),
        linker=AttachmentPoint(attachables={2, 3}),
        bondtype=BondType.DOUBLE,
    )

    # derived Connectors
    node_0_conn_0 = conn_type_a.copy()
    node_0_conn_1 = conn_type_a.copy()

    node_1_conn_0 = conn_type_a.counterpart()
    node_1_conn_1 = conn_type_b.copy()

    node_2_conn_0 = conn_type_a.counterpart()
    node_2_conn_1 = conn_type_b.copy()
    
    return LinkerTestExample(
        cycle_graph(3),
        connector_assignments={
            0 : (node_0_conn_0, node_0_conn_1),
            1 : (node_1_conn_0, node_1_conn_1),
            2 : (node_2_conn_0, node_2_conn_1),
        },
        possible_solutions=(
            {
                (0, 1) : (node_0_conn_0, node_1_conn_0),
                (1, 2) : (node_1_conn_1, node_2_conn_1),
                (0, 2) : (node_0_conn_1, node_2_conn_0),
            },
            {
                (0, 1) : (node_0_conn_1, node_1_conn_0),
                (1, 2) : (node_1_conn_1, node_2_conn_1),
                (0, 2) : (node_0_conn_0, node_2_conn_0),
            },
        )
    )
    
# tests proper
@pytest.mark.parametrize(
    'linker_example',
    [
        triangle_example(),
        # pytest.param(
        #     ...,
        #     marks=pytest.mark.xfail(
        #         raises=EdgeMissingError,
        #         reason='No complete pairing possible for the Connector assigned to this topology',
        #         strict=True,
        #     )
        # ),
    ]
)
def test_linker_examples(linker_example : LinkerTestExample) -> None:
    '''
    Test that the linker routine correctly finds a pairing solution in examples
    where one exists, and correctly rejects examples in which one doesn't
    '''
    conn_map = deduce_connections_from_topology(
        linker_example.topology,
        mapped_connectors=linker_example.connector_assignments,
    ) # will raise EdgeMissingError if solution is not found
    
    assert conn_map in linker_example.possible_solutions