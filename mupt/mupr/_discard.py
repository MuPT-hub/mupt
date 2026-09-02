'''Vestiges from pre-refactor code which will be discarded, 
but haven't been fully been scrapped for parts out yet'''

from typing import Iterable, AbstractSet
from networkx import Graph

from .linking import BijectionError
from .connection.connectors import Connector
from .connection.exceptions import IncompatibleConnectorError

Connection = tuple[Connector, Connector]


# Validators - TB: absorb useful parts and discard these as part of refactor
def check_connections_compatible_with_primitive_registry(
    primitive_registry : UniqueRegistry['PrimitiveHandle', 'Primitive'],
    connections : Iterable[Connection], # DEV: weakened type requirement here, even though in practice this will most like be a set or frozenset
) -> None:
    '''
    Check that a collection of connections (i.e. pairs of (PrimitiveHandle, ConnectorAddress) references)
    is absolutely compatible with a handled registry of Primitives
    '''
    for (prim_handle_1, conn_addr_1), (prim_handle_2, conn_addr_2) in connections:
        if prim_handle_1 == prim_handle_2:
            raise ValueError(f'Attempted to connect Primitive with handle "{prim_handle_1}" to itself')
        
        if conn_addr_1 == conn_addr_2:
            raise IncompatibleConnectorError(f'Connections must be between distinct pair of Connector instances, not single Connector at address {conn_addr_1}')
        
        for prim_handle in (prim_handle_1, prim_handle_2):
            if prim_handle not in primitive_registry:
                raise ValueError(f'Primitive with handle "{prim_handle}" referenced in internal connections but does not exist in provided registry of children')
            
        if not Connector.bondable_with( # NOTE: fetch also implicitly checks each Connector exists on respective child
            primitive_registry[prim_handle_1].connector(conn_addr_1),
            primitive_registry[prim_handle_2].connector(conn_addr_2),
        ):
            raise IncompatibleConnectorError(
                f'Connector {conn_addr_1} on Primitive {prim_handle_1} is not bondable with Connector {conn_addr_2} on Primitive {prim_handle_2}'
            )

def check_primitive_registry_bijective_to_topology_nodes(
    primitive_registry : UniqueRegistry['PrimitiveHandle', 'Primitive'],
    topology : Graph,
) -> None:
    '''
    Verify 1:1 correspondence between the reference handles in a 
    registry of Primitives and the nodes in an incidence topology
    '''
    num_children : int = len(primitive_registry) # perform cheap counting check first to fail faster
    if topology.number_of_nodes() != num_children:
        raise BijectionError(f'Cannot bijectively map {num_children} child Primitives onto {topology.number_of_nodes()}-element topology')
    
    node_labels = set(topology.nodes)
    child_handles = set(primitive_registry.keys())
    if node_labels != child_handles:
        raise BijectionError(
            f'Set underlying topology does not correspond to handles on child Primitives; {len(node_labels - child_handles)} element(s)'\
            f' present without associated children, and {len(child_handles - node_labels)} child Primitive(s) are unrepresented in the topology'
        )

def check_connections_bijectiLve_to_topology_edges(
    connections : AbstractSet[Connection],
    topology : Graph,
) -> None:
    '''
    Verify that a 1:1 correspondence exists between the internal connections
    (Connectors paired between sibling child Primitives) and the edges present in the incidence topology
    '''
    num_connections : int = len(connections) # perform cheap counting check first to fail faster
    if (num_edges := topology.number_of_edges()) != num_connections:
        raise BijectionError(f'Cannot bijectively map {num_connections} internal connections onto {num_edges}-edge topology')

    edge_labels = set(frozenset(edge) for edge in topology.edges) # cast to frozenset to remove order-dependence
    if edge_labels != connections:
        raise BijectionError(
            f'Incident pairs in associated topology do not correspond to internally-connected pairs of child Primitives;'\
            f'{len(edge_labels - connections)} edge(s) have no corresponding connection, '\
            f'and {len(connections - edge_labels)} internal connection(s) are unrepresented in the topology'
        )
