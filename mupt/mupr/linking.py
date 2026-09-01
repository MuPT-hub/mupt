'''
Utilities for linking Connectors to form two-way bonded connections
in a MuPT representation based on global topology specification
'''

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    AbstractSet,
    Callable,
    Collection,
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

DEFAULT_ITER_RULE : Callable[[int], int] = lambda graph_size : 10*graph_size # TB DEV: 10 is just a number I made up :P

def deduce_connections_from_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T], indicating node type
    mapped_connectors : Mapping[T, Collection[Connector]], # Collection (rather than Iterable) needed for length check
    n_iter_max_rule : Callable[[int], int]=DEFAULT_ITER_RULE, 
) -> Mapping[tuple[T, T], Mapping[T, Connector]]:
    """
    Given a connectivity graph and a collection of ConnectorManagers
    mapped to a (non-proper) subset of the nodes of that graph,
    deduces if it is possible to connect the Connectors within those managers
    along the edges of the graph, and if so returns an explicit mapping of those connections

    If pairing is impossible, will raise Exception instead
    """
    if not set(topology.nodes).issubset(set(mapped_connectors.keys())): 
        # Weaker size requirement; nodes need not be in 1:1 correspondence with Connector collections, merely covered by them
        raise NodeMappingError('Not all nodes in the given topology are convered by collections of Connectors')
    
    for node in topology.nodes:
        if (num_connectors := len(mapped_connectors[node])) < (num_neighbors := topology.degree[node]):
            raise NodeMappingError(
                f'Node {node!r} has {num_neighbors} neighbors, but only'
                f'{num_connectors} connection to distribute among them'
            )

    # working with EQUIVALENCE CLASSES of Connectors, rather than connectors directly
    # pares down cartesian product for search and makes unique-choice condition less stringent
    conn_equiv_classes : dict[T, set[set[Connector]]] = {
        node_label : set(
            set(equiv_class)
                for equiv_class in equivalence_classes(
                    connectors,
                    relation=Connector.fungible_with,
                )
        )
        for node_label, connectors in mapped_connectors.items() 
    }

    num_total_edges : int = topology.number_of_edges()
    unpaired_edges : set[tuple[T, T]] = set(topology.edges)
    connection_map : Mapping[tuple[T, T], Mapping[T, Connector]] = dict()

    n_iter : int = 0
    n_iter_max : int = n_iter_max_rule(topology.number_of_nodes())
    while (n_iter < n_iter_max) and unpaired_edges:
        n_paired_new : int = 0
        unpaired_updated = set()
        
        for edge_labels in unpaired_edges:
            node_label_former, node_label_latter = edge_labels
            conn_classes_former : set[set[Connector]] = conn_equiv_classes[node_label_former]
            conn_classes_latter : set[set[Connector]] = conn_equiv_classes[node_label_latter]
                
            pair_choice_ambiguous : bool = False
            chosen_connector_classes : Optional[tuple[set[Connector], set[Connector]]] = None

            for conn_class_former, conn_class_latter in cartesian(
                conn_classes_former,
                conn_classes_latter,
            ):
                # one pair from product of equiv classes bondable ==> any pair bondable
                if not Connector.bondable_with(
                    arbitrary_element(conn_class_former),
                    arbitrary_element(conn_class_latter),
                ):
                    continue

                if (chosen_connector_classes is None):
                    # take note of first compatible pair found
                    chosen_connector_classes = (conn_class_former, conn_class_latter) 
                else:
                    pair_choice_ambiguous = True 
                    break # further search can't disambiguate choice, stop early to save computation
                # TB TODO: make record of all choice when ambiguous, provide way to pass heuristic for breaking ties
                
            if pair_choice_ambiguous:
                LOGGER.debug(f'Choice of Connector pair ambiguous for edge {edge_labels}, skipping')
                unpaired_updated.add(edge_labels) # "try again next time!"
                # NB: opting to collected unmatched edges (rather than popping
                # matched ones) to avoid modifying set while iterating over it
                continue
            elif (chosen_connector_classes is None):
                raise EdgeMissingError(f'No compatible Connector pairs found for edge {edge_labels}')

            # if unambiguous pairing is present, draw representatives of respective compatible classes and bind them
            # TODO: mark off newly-connected Connectors from candidates for bondability
            conn_class_former, conn_class_latter = chosen_connector_classes # TODO: avoid redudant unpacking
            connection_map[edge_labels] = {
                node_label_former : conn_class_former.pop(),
                node_label_latter : conn_class_latter.pop(),
            }
            if not conn_class_former:
                conn_classes_former.remove(conn_class_former) # delete class if emptied
            
            if not conn_class_latter:
                conn_classes_latter.remove(conn_class_latter) # delete class if emptied
                    
            n_paired_new += 1
        
        # tee up next iteration;
        unpaired_edges = unpaired_updated
        n_iter += 1
        LOGGER.info(
            f'Paired up {n_paired_new} new edges after {n_iter} iteration(s);'
            '{len(unpaired_edges)}/{num_total_edges} edges remain unpaired'
        )
        
        ## halt if no further connections can be made
        if n_paired_new == 0:
            LOGGER.info(f'No new edges paired, halting registration loop')
            break 
        
    if any(unpaired_edges):
        raise EdgeMissingError(f'Could not identify connection for every edge; try running registration procedure for >{n_iter_max} iterations, or check topology/Connectors')
    
    return connection_map

def assign_connections_from_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T]
    mapped_connectors : Mapping[T, ConnectorManager],
    n_iter_max_rule : Callable[[int], int]=DEFAULT_ITER_RULE,
) -> None:
    """Deduce connections from graph and mapped ConnectorManagers and assign neighborship based on it"""
    connections : Mapping[tuple[T, T], Mapping[T, Connector]] = deduce_connections_from_topology(
        topology,
        mapped_connectors=mapped_connectors,
        n_iter_max_rule=n_iter_max_rule,
    )
    
    for (node_former, node_latter), connector_map in connections.items():
        connector_map[node_former].neighbor = connector_map[node_latter]
