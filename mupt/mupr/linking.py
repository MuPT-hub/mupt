'''
Utilities for linking Connectors to form two-way bonded connections
in a MuPT representation based on global topology specification
'''

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    overload,
)
T = TypeVar('T')

from itertools import product as cartesian

from networkx import Graph
from networkx.utils import arbitrary_element
from networkx.algorithms import equivalence_classes

from .connection.connectors import Connector
from .connection.exceptions import (
    IncompatibleConnectorError,
    MissingConnectorError,
    UnboundConnectorError,
)


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


DEFAULT_ITER_RULE : Callable[[int], int] = lambda graph_size : 10*graph_size # TB DEV: 10 is just a number I made up :P

def _check_connectors_cover_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T], indicating node type
    mapped_connectors : Mapping[T, Collection[Connector]], # Collection (rather than Iterable) needed for length check
) -> None:
    '''
    Necessary (but not sufficient) conditions to ensure a map from
    graph nodes to collections of Connectors covers all nodes and edges
    
    Specifically, checks that:
    * Preimage of map contains node set (i.e. every node gets some collection of Connectors)
    * Image of each node has no fewer Connectors than the node has neighbors
    
    Returns silently if passing; raises NodeMappingError otherwise
    '''
    if not set(topology.nodes).issubset(set(mapped_connectors.keys())): 
        # Weaker size requirement; nodes need not be in 1:1 correspondence with Connector collections, merely covered by them
        raise NodeMappingError('Not all nodes in the given topology are convered by collections of Connectors')
    
    for node in topology.nodes:
        if (num_connectors := len(mapped_connectors[node])) < (num_neighbors := topology.degree[node]):
            raise NodeMappingError(
                f'Node {node!r} has {num_neighbors} neighbors, but only'
                f'{num_connectors} connection to distribute among them'
            )
            
# TODO: ensure no ambiguity arises on deduction over parallel MultiGraph edges 
def deduce_connections_from_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T], indicating node type
    mapped_connectors : Mapping[T, Collection[Connector]], # Collection (rather than Iterable) needed for length check
    n_iter_max_rule : Optional[Callable[[int], int]]=None, 
) -> Mapping[tuple[T, T], Mapping[T, Connector]]:
    """
    Given a connectivity graph and a collection of ConnectorManagers
    mapped to a (non-proper) subset of the nodes of that graph,
    deduces if it is possible to connect the Connectors within those managers
    along the edges of the graph, and if so returns an explicit mapping of those connections

    If pairing is impossible, will raise Exception instead
    """
    _check_connectors_cover_topology(topology, mapped_connectors)
    if n_iter_max_rule is None:
        n_iter_max_rule = DEFAULT_ITER_RULE
    
    # working with EQUIVALENCE CLASSES of Connectors, rather than connectors directly
    # pares down cartesian product for search and makes unique-choice condition less stringent
    conn_equiv_classes : dict[T, set[frozenset[Connector]]] = {
        node_label : equivalence_classes(connectors, relation=Connector.fungible_with)
            for node_label, connectors in mapped_connectors.items() 
    }
    unpaired_edges : set[tuple[T, T]] = set(topology.edges)
    connection_map : Mapping[tuple[T, T], Mapping[T, Connector]] = dict()

    n_iter : int = 0
    n_iter_max : int = n_iter_max_rule(topology.number_of_nodes())
    while (n_iter < n_iter_max) and unpaired_edges:
        n_paired_new : int = 0
        unpaired_updated = set()
        
        for edge_labels in unpaired_edges:
            node_label_former, node_label_latter = edge_labels
            conn_classes_former : set[frozenset[Connector]] = conn_equiv_classes[node_label_former]
            conn_classes_latter : set[frozenset[Connector]] = conn_equiv_classes[node_label_latter]
                
            pair_choice_ambiguous : bool = False
            chosen_connectors : Optional[dict[T, Connector]] = None

            for conn_class_former, conn_class_latter in cartesian(
                conn_classes_former,
                conn_classes_latter,
            ):
                # one pair from product of equiv classes bondable ==> any pair bondable
                peek_conn_former = arbitrary_element(conn_class_former)
                peek_conn_latter = arbitrary_element(conn_class_latter)
                
                if not Connector.bondable_with(peek_conn_former, peek_conn_latter):
                    continue
                elif (chosen_connectors is None): # take note of first compatible pair found
                    chosen_connectors = {
                        node_label_former : peek_conn_former,
                        node_label_latter : peek_conn_latter,
                    }
                else: # if compatible classes were found previously, choice is ambiguous; halt class assessment
                    # TB TODO: provide means to break ties when ALL edge pairings are ambiguous (keep record, rather than halting)
                    pair_choice_ambiguous = True 
                    break # further search can't disambiguate choice, stop early to save computation
                
            # Decide how to continue after equivalence classes have been assessed
            if (chosen_connectors is None):
                raise EdgeMissingError(f'No compatible Connector pairs found for edge {edge_labels}')
            
            if pair_choice_ambiguous:
                LOGGER.debug(f'Choice of Connector pair ambiguous for edge {edge_labels}, skipping')
                unpaired_updated.add(edge_labels) # "try again next time!"
                # NB: opting to collected unmatched edges (rather than popping
                # matched ones) to avoid modifying set while iterating over it
                continue

            # Pairing is unambiguous; mark off chosen representatives and update their equivalence classes if necessary
            for equiv_classes, equiv_class, representative in (
                (conn_classes_former, conn_class_former, peek_conn_former),
                (conn_classes_latter, conn_class_latter, peek_conn_latter),
            ):
                equiv_classes.remove(equiv_class)
                equiv_class -= {representative}
                if equiv_class: # re-add part only if it is non-empty after the pairing
                    equiv_classes.add(equiv_class)
            
            # Lock in pair of Connectors and proceed
            connection_map[edge_labels] = chosen_connectors
            n_paired_new += 1
        
        # tee up next iteration;
        unpaired_edges = unpaired_updated
        n_iter += 1
        LOGGER.info(
            f'Paired up {n_paired_new} new edges after {n_iter} iteration(s);'
            '{len(unpaired_edges)}/{num_total_edges} edges remain unpaired'
        )
        
        # halt if no further connections can be made
        if n_paired_new == 0:
            LOGGER.info(f'No new edges paired, halting registration loop')
            break 
        
    if any(unpaired_edges):
        raise EdgeMissingError(f'Could not identify connection for every edge; try running registration procedure for >{n_iter_max} iterations, or check topology/Connectors')
    
    return connection_map

def assign_connections_from_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T]
    mapped_connectors : Mapping[T, Collection[Connector]],
    n_iter_max_rule : Optional[Callable[[int], int]]=None,
) -> None:
    """Deduce connections from graph and mapped ConnectorManagers and assign neighborship based on it"""
    connections : Mapping[tuple[T, T], Mapping[T, Connector]] = deduce_connections_from_topology(
        topology,
        mapped_connectors=mapped_connectors,
        n_iter_max_rule=n_iter_max_rule,
    )
    
    for (node_former, node_latter), connector_map in connections.items():
        connector_map[node_former].neighbor = connector_map[node_latter]
