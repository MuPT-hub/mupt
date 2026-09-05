'''
Utilities for linking Connectors to form two-way bonded connections
in a MuPT representation based on global topology specification
'''

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Callable,
    Collection,
    Mapping,
    Optional,
    TypeVar,
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


DEFAULT_ITER_RULE : Callable[[int], int] = lambda graph_size : 10*graph_size # TB DEV: 10 is just a sensible number I made up :P

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
            
def deduce_connections_from_topology(
    topology : Graph, # TB: Graph[T], indicating node type
    mapped_connectors : Mapping[T, Collection[Connector]],
    n_iter_max_rule : Optional[Callable[[int], int]]=None, 
) -> Mapping[tuple[T, T], tuple[Connector, Connector]]:
    """
    Given a connectivity graph and a collection of ConnectorManagers
    mapped to a (non-proper) subset of the nodes of that graph,
    deduces if it is possible to pair those Connectors along the edges of the graph,
    and if so returns an explicit mapping of those connections

    Returned mapping maps edges, as ordered 2-tuples of nodes (i.e. (a, b) w/ a < b),
    to 2-tuples of the Connectors associated to that same edge and in the same order
    E.g. (0, 1) : (<Connector on 0>, <Connector on 1>)

    If pairing is impossible, will raise Exception instead
    """
    # TODO: ensure no ambiguity arises on deduction over parallel MultiGraph edges 
    _check_connectors_cover_topology(topology, mapped_connectors)
    if n_iter_max_rule is None:
        # set here (rather than as arg default) so external callers can be oblivious to default and just use None
        n_iter_max_rule = DEFAULT_ITER_RULE 
    
    # working with EQUIVALENCE CLASSES of Connectors, rather than connectors directly
    # pares down cartesian product for search and makes unique-choice condition less stringent
    #
    # Equivalence relations (in this case, Connector fungibility) naturally induce partitions
    # (see https://en.wikipedia.org/wiki/Equivalence_relation#Fundamental_theorem_of_equivalence_relations)
    conn_partitions : dict[T, set[frozenset[Connector]]] = {
        node_label : equivalence_classes(connectors, relation=Connector.fungible_with)
            for node_label, connectors in mapped_connectors.items() 
    }
    num_total_edges : int = topology.number_of_edges()
    unpaired_edges : set[tuple[T, T]] = set(topology.edges)
    connection_map : Mapping[tuple[T, T], tuple[Connector, Connector]] = dict()

    n_iter : int = 0
    n_iter_max : int = n_iter_max_rule(topology.number_of_nodes())
    while (n_iter < n_iter_max) and unpaired_edges:
        LOGGER.debug(f'Beginning Connector linking iteration {n_iter}:')
        n_paired_new : int = 0
        unpaired_updated = set()
        
        # TB TODO: add option to introduce some stochasticity for discovering alternate solutions
        for edge_labels in unpaired_edges: 
            node_label_former, node_label_latter = edge_labels
            LOGGER.debug(f'Attempting to find compatible Connectors for edge {edge_labels}:')
                
            # NB: assigning to vars, rather than referencing directly in 
            # cartesian(), as refs are needed later for updating seen partitions 
            conn_partition_former : set[frozenset[Connector]] = conn_partitions[node_label_former]
            conn_partition_latter : set[frozenset[Connector]] = conn_partitions[node_label_latter]
            
            pair_choice_ambiguous : bool = False
            chosen_connectors : Optional[tuple[Connector, Connector]] = None
            
            # Screen equivalence classes to see if 0, 1, or many matches are present
            for conn_part_former, conn_part_latter in cartesian(
                conn_partition_former,
                conn_partition_latter,
            ):
                peek_conn_former = arbitrary_element(conn_part_former)
                peek_conn_latter = arbitrary_element(conn_part_latter)
                LOGGER.debug(f'Examining Connector pair {peek_conn_former!r} and {peek_conn_latter!r}')
                
                if not Connector.bondable_with(peek_conn_former, peek_conn_latter):
                    # any pair from the product of equivalence classes being bondable implies any pair is
                    LOGGER.debug(f'Found pair to be incompatible, continuing...')
                    continue
                
                if not chosen_connectors: # take note of first compatible pair found
                    LOGGER.debug(f'Chosen pair is a match!')
                    chosen_connectors = (peek_conn_former, peek_conn_latter)
                    break
                else: 
                    LOGGER.debug(f'Choice of Connector pair ambiguous for edge {edge_labels}, skipping')
                    pair_choice_ambiguous = True 
                    # TB TODO: provide means to break ties when ALL edge pairings
                    # are ambiguous (keep record, rather than halting)
                    break
                
            # Decide how to continue after equivalence classes have been assessed
            if (chosen_connectors is None):
                raise EdgeMissingError(f'No compatible Connector pairs found for edge {edge_labels}')
            
            if pair_choice_ambiguous:
                unpaired_updated.add(edge_labels) # "try again next time!"
                # NB: opting to collected unmatched edges (rather than popping
                # matched ones) to avoid modifying set while iterating over it
                continue
            else:
                # If unambiguous, record chosen representatives, mark used up, 
                # and update their equivalence classes if emptied
                LOGGER.debug('Updating explored parts of partitions')
                connection_map[edge_labels] = chosen_connectors
                for partition, part, representative, descriptor in (
                    (conn_partition_former, conn_part_former, peek_conn_former, 'former'),
                    (conn_partition_latter, conn_part_latter, peek_conn_latter, 'latter'),
                ):
                    partition.remove(part)
                    part -= {representative}
                    if part:
                        partition.add(part)
                    else:
                        LOGGER.debug(
                            'Examined part has been emptied and removed from '
                            f'{descriptor} partition; {len(partition)} parts remain'
                        )
                n_paired_new += 1
        
        # tee up next iteration;
        unpaired_edges = unpaired_updated
        n_iter += 1
        LOGGER.info(
            f'Paired up {n_paired_new} new edges after {n_iter} iteration(s); '
            f'{len(unpaired_edges)}/{num_total_edges} edges remain unpaired'
        )
        
        if n_paired_new == 0:
            LOGGER.info(f'No new edges paired, halting registration loop')
            break 
        
    if any(unpaired_edges):
        raise EdgeMissingError(
            f'Could not identify connection for every edge; try running registration '
            'procedure for >{n_iter_max} iterations, or check topology/Connectors'
        )
    else:
        LOGGER.info(
            f'Linking protocol successful! {num_total_edges - len(unpaired_edges)}/{num_total_edges} edges were assigned Connector pairings'
        )
    
    return connection_map

def assign_connections_from_topology(
    topology : Graph, # TB: if Graph supported Generic subscripting, this annotation would be Graph[T]
    mapped_connectors : Mapping[T, Collection[Connector]],
    n_iter_max_rule : Optional[Callable[[int], int]]=None,
) -> None:
    """Deduce connections from graph and mapped ConnectorManagers and assign neighborship based on it"""
    connections = deduce_connections_from_topology(
        topology,
        mapped_connectors=mapped_connectors,
        n_iter_max_rule=n_iter_max_rule,
    )
    
    for (node_former, node_latter), (conn_former, conn_latter) in connections.items():
        conn_former.neighbor = conn_latter
