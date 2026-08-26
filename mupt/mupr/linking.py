'''
Utilities for linking Connectors to form two-way bonded connections
in a MuPT representation based on global topology specification
'''

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    AbstractSet,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
    overload,
    TYPE_CHECKING
)
T = TypeVar('T')

from dataclasses import dataclass
from itertools import product as cartesian

from networkx import Graph
from networkx.utils import arbitrary_element
from networkx.algorithms import equivalence_classes

if TYPE_CHECKING: # TODO: figure out how to non-circularly import even w/o typechecking
    from .primitives import Primitive, PrimitiveLabel, PrimitiveHandle

from .connection.connectors import Connector
from .connection.types import ConnectorAddress, ConnectorHandle
from .connection.management import ConnectorManager
from .connection.exceptions import (
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)
Connection = tuple[Connector, Connector]
from ..mutils.containers import UniqueRegistry


class BijectionError(ValueError):
    '''Raised when a pair of objects expected to be in 1-to-1 correspondence are mismatched'''
    ...

class GraphLinkingError(ValueError):
    '''Raised when an invalid mapping to a graph is encountered'''
    ...

class NodeMappingError(GraphLinkingError):
    '''Raised when an invalid mapping between an object and a graph node is encountered'''
    ...

class EdgeMissingError(GraphLinkingError):
    '''Raised when an invalid mapping between a pair of objects and a graph edge is encountered'''
    ...

# Validators
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

def check_connections_bijective_to_topology_edges(
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


# Deductions of connections from graphs
@dataclass(frozen=True) # needed for hashability
class ConnectorReference: # TB TODO: deprecate code which depends on this before final merge
    '''Lightweight reference to a Connector on a Primitive, identified by the Primitive's handle and the Connector's handle'''
    primitive_handle : 'PrimitiveHandle'
    connector_handle : ConnectorHandle  
    
    def with_reassigned_primitive(self, new_primitive_handle : 'PrimitiveHandle') -> 'ConnectorReference':
        '''Return a copy of this ConnectorReference with a different PrimitiveHandle'''
        return ConnectorReference(
            primitive_handle=new_primitive_handle,
            connector_handle=self.connector_handle,
        )
        
    def __str__(self) -> str:
        return f'Connector "{self.connector_handle}" attached to Primitive "{self.primitive_handle}"'
    

# def check_connector_map_compatible_with_topology() -> bool:
#     ...

def deduce_connections_from_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T]
    mapped_connectors : Mapping[T, ConnectorManager],
    n_iter_max_rule : Callable[[int], int]=lambda graph_size : 10*graph_size, # TB DEV: 10 is just a number I made up :P
) -> Mapping[tuple[T, T], Mapping[T, ConnectorAddress]]:
    """
    Given a connectivity graph and a collection of ConnectorManagers
    mapped to a (non-proper) subset of the nodes of that graph,
    deduces if it is possible to connect the Connectors within those managers
    along the edges of the graph, and if so returns an explicit mapping of those connections

    If pairing is impossible, will raise Exception instead
    """
    if not set(topology.nodes).issubset(set(mapped_connectors.keys())): 
        # weaker requirement of containing (rather than being equal) to vertex set suffices
        raise NodeMappingError('Not all nodes in the given topology are convered by collections of Connectors')

    num_total_edges : int = topology.number_of_edges()
    unpaired_edges : set[tuple[T, T]] = set(topology.edges)
    connection_map : Mapping[tuple[T, T], Mapping[T, ConnectorAddress]] = dict()

    # Begin iterative pairing logic
    n_iter : int = 0
    n_iter_max : int = n_iter_max_rule(topology.number_of_nodes())
    while (n_iter < n_iter_max) and unpaired_edges:
        n_paired_new : int = 0
        unpaired_updated = set()
        
        for edge_labels in unpaired_edges:
            node_label_former, node_label_latter = edge_labels
            conn_mgr_former = mapped_connectors[node_label_former].connectors
            conn_mgr_latter = mapped_connectors[node_label_latter].connectors
                
            ## attempt to identify if there is a UNIQUE pair of bondable classes of Connectors along the edge
            pair_choice_ambiguous : bool = False
            chosen_connectors : Optional[tuple[Connector, Connector]] = None
            conns_fitting_edge = Connector.bondable_connector_pairs(conn_mgr_former, conn_mgr_latter)

            for (conn_former, conn_latter) in conns_fitting_edge:
                # assert Connector.bondable_with(conn_former, conn_latter) # redundant check for the paranoid
                if compatible_class_labels is None:
                    compatible_class_labels = (class_label1, class_label2) # take note of first compatible pair found
                else:
                    # TB TODO: modify to work when there are TOO MANY options (e.g. none of remaining unpaired have unique choice)
                    pair_choice_ambiguous = True 
                    break # further search can't disambiguate choice, stop early to save computation
                
            if pair_choice_ambiguous:
                LOGGER.debug(f'Choice of Connector pair ambiguous for edge {edge_labels}, skipping')
                unpaired_updated.add(edge_labels) # "try again next time!"
                continue
            elif (compatible_class_labels is None):
                raise EdgeMissingError(f'No compatible Connector pairs found for edge {edge_labels}')

            ## if unambiguous pairing is present, draw representatives of respective compatible classes and bind them
            # TODO: mark off newly-connected Connectors from candidates for bondability
            connection_map[edge_labels] = {
                node_label_former : conn_former,
                node_label_latter : conn_latter
            }
            n_paired_new += 1
        
        ## tee up next iteration; halt if no further connections can be made
        unpaired_edges = unpaired_updated
        n_iter += 1
        
        LOGGER.info(f'Paired up {n_paired_new} new edges after {n_iter} iteration(s); {len(unpaired_edges)}/{num_total_edges} edges remain unpaired')
        if n_paired_new == 0:
            LOGGER.info(f'No new edges paired, halting registration loop')
            break 
        
    if any(unpaired_edges):
        raise EdgeMissingError(f'Could not identify connection for every edge; try running registration procedure for >{n_iter_max} iterations, or check topology/Connectors')
    
    return connection_map

def assign_connections_from_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T]
    mapped_connectors : Mapping[T, ConnectorManager],
    n_iter_max_rule : Callable[[int], int]=lambda graph_size : 10*graph_size, # TB DEV: 10 is just a number I made up :P
) -> None:
    """Deduce connections from graph and mapped ConnectorManagers and assign neighborship based on it"""
    ...
