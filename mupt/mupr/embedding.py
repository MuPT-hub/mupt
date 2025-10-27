'''Utilities for verifying (and producing) relationships between Topologies and other MuPT core components'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

# DEVNOTE: this is not a submodule under topology to avoid circular imports
# and to shelter MID Graphs from needing to know about HOW they're embedded

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
    overload,
)
T = TypeVar('T')
PrimitiveLabel = TypeVar('PrimitiveLabel', bound=Hashable)
PrimitiveHandle = tuple[PrimitiveLabel, int] # (label, uniquification index)

from dataclasses import dataclass
from itertools import product as cartesian

from networkx import Graph
from networkx.utils import arbitrary_element
from networkx.algorithms import equivalence_classes

# DEV: this module CANNOT import Primitive if circular imports are to be avoided
from .connection import Connector, ConnectorHandle
from .topology import TopologicalStructure


class GraphEmbeddingError(ValueError):
    '''Raised when an invalid mapping to a graph is encountered'''
    ...

class NodeEmbeddingError(GraphEmbeddingError):
    '''Raised when an invalid mapping between an object and a graph node is encountered'''
    ...

class EdgeEmbeddingError(GraphEmbeddingError):
    '''Raised when an invalid mapping between a pair of objects and a graph edge is encountered'''
    ...


def mapped_equivalence_classes(
        objects : Iterable[T],
        relation : Callable[[T, T], bool],
    ) -> dict[Hashable, list[T]]:
    """
    Partition a collection of objects into equivalence classes by
    an equivalence relation defined on pairs of those objects
    
    Return dict whose values are the equivalence classes and 
    whose keys are unique labels for each class
    """
    # DEV: more-or-less reimplements networkx's equivalence_classes but w/o the frozenset collapsing at the end - find way to incorporate going forward
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.minors.equivalence_classes.html
    equiv_classes : list[list[T]] = []
    for obj in objects:
        for equiv_class in equiv_classes:
            if relation(obj, arbitrary_element(equiv_class)):
                equiv_class.append(obj)
                break
        else:
            equiv_classes.append([obj])
    
    return {
        i : equiv_class # DEV: opting for index as default unique label for now; eventually want labels to be semantically-related to each class
            for i, equiv_class in enumerate(equiv_classes)
    }

@dataclass(frozen=True) # needed for hashability
class ConnectorReference:
    '''Lightweight reference to a Connector on a Primitive, identified by the Primitive's handle and the Connector's handle'''
    primitive_handle : PrimitiveHandle
    connector_handle : ConnectorHandle  
    
    def with_reassigned_primitive(self, new_primitive_handle : PrimitiveHandle) -> 'ConnectorReference':
        '''Return a copy of this ConnectorReference with a different PrimitiveHandle'''
        return ConnectorReference(
            primitive_handle=new_primitive_handle,
            connector_handle=self.connector_handle,
        )
        
    def __str__(self) -> str:
        return f'Connector "{self.connector_handle}" attached to Primitive "{self.primitive_handle}"'
    
@overload
def flexible_connector_reference(
    primitive_handle : PrimitiveHandle,
    connector_handle : ConnectorHandle,
) -> ConnectorReference: 
    ...
    
@overload
def flexible_connector_reference(
    primitive_handle : ConnectorReference,
) -> ConnectorReference:
    ...

def flexible_connector_reference(
    primitive_handle : Union[PrimitiveHandle, ConnectorReference],
    connector_handle : Optional[ConnectorHandle]=None,
) -> ConnectorReference:
    '''Utility to interchangeably handle cases of passing a (PrimitiveHandle, ConnectorHandle) pair or a ConnectorReference'''
    if isinstance(primitive_handle, ConnectorReference):
        if connector_handle is not None:
            raise ValueError('If passing a ConnectorReference as the first argument, the second argument must be omitted')
        return primitive_handle
    elif connector_handle is None:
        raise ValueError('If passing a PrimitiveHandle as the first argument, the second argument (ConnectorHandle) must be provided')
    else:
        return ConnectorReference(
            primitive_handle=primitive_handle,
            connector_handle=connector_handle,
        )
    
    
def infer_connections_from_topology(
    topology : TopologicalStructure,
    mapped_connectors : Mapping[PrimitiveHandle, Mapping[ConnectorHandle, Connector]],
    n_iter_max : int=25, # DEV: this is just a number I made up :P
) -> dict[frozenset[PrimitiveHandle], frozenset[ConnectorReference]]:
    """
    Deduce if a collection of Connectors associated to each node in a topology
    can be identified with the edges in that topology, such that each pair of Connectors is bondable
    
    Returns a first mapping of pairs of node labels (one pair for each edge)
    to a mapping from node labels to the Connector associated to that edge,
    and a second mapping of node labels to remaining external Connectors, if any remain unpaired
    
    If pairing is impossible, will raise Exception instead
    """
    if not set(topology.nodes).issubset(set(mapped_connectors.keys())): 
        # weaker requirement of containing (rather than being equal) to vertex set suffices
        raise NodeEmbeddingError('Connector collection labels do not match topology node labels')

    # Initialize containers for tracking pairing progress
    num_total_edges : int = topology.number_of_edges()
    unpaired_edges : set[frozenset[PrimitiveHandle]] = set(frozenset(edge) for edge in topology.edges)
    paired_connectors : dict[frozenset[PrimitiveHandle], frozenset[ConnectorReference]] = {}
    
    connector_equiv_classes : dict[PrimitiveHandle, dict[int, set[ConnectorHandle]]] = {}
    for owner_handle, connector_map in mapped_connectors.items():
        equiv_class = equivalence_classes(
            connector_map,
            relation=lambda conn_handle1, conn_handle2 : Connector.fungible_with(
                connector_map[conn_handle1],
                connector_map[conn_handle2],
            ),
        )
        connector_equiv_classes[owner_handle] = {
            i : set(equiv_class)
                for i, equiv_class in enumerate(equiv_class)
        }

    # iteratively pair connectors along edges
    n_iter : int = 0
    while (n_iter < n_iter_max) and unpaired_edges:
        n_paired_new : int = 0
        unpaired_updated = set()
        
        for edge_labels in unpaired_edges:
            owner_handle1, owner_handle2 = edge_labels    
            ## attempt to identify if there is a UNIQUE pair of bondable classes of Connectors along the edge
            pair_choice_ambiguous : bool = False
            compatible_class_labels : Optional[tuple[Connector, Connector]] = None
            for (class_label1, eq_class_1), (class_label2, eq_class_2) in cartesian(
                connector_equiv_classes[owner_handle1].items(),
                connector_equiv_classes[owner_handle2].items(),
            ):
                if not Connector.bondable_with(
                    mapped_connectors[owner_handle1][arbitrary_element(eq_class_1)],
                    mapped_connectors[owner_handle2][arbitrary_element(eq_class_2)],                               
                ): 
                    continue # skip over incompatible Connector classes
                
                if compatible_class_labels is None:
                    compatible_class_labels = (class_label1, class_label2) # take note of first compatible pair found
                else:
                    pair_choice_ambiguous = True
                    break # further search can't disambiguate choice, stop early to save computation
                
            if pair_choice_ambiguous:
                LOGGER.debug(f'Choice of Connector pair ambiguous for edge {edge_labels}, skipping')
                unpaired_updated.add(edge_labels) # "try again next time!"
                continue
            elif (compatible_class_labels is None):
                raise EdgeEmbeddingError(f'No compatible Connector pairs found for edge {edge_labels}')

            ## if unambiguous pairing is present, draw representatives of respective compatible classes and bind them
            chosen_representatives : set[ConnectorReference] = set()
            for (class_label, owner_label) in zip(compatible_class_labels, edge_labels):
                equiv_class = connector_equiv_classes[owner_label][class_label]
                chosen_representatives.add(
                    ConnectorReference(
                        primitive_handle=owner_label,
                        connector_handle=equiv_class.pop(), # DEV: index here shouldn't matter, but will standardized to match arbitrary element selection
                    )
                )
                if len(equiv_class) == 0: # remove bin from equivalence classes if empty after drawing
                    _ = connector_equiv_classes[owner_label].pop(class_label)
            paired_connectors[frozenset(edge_labels)] = frozenset(chosen_representatives)
            n_paired_new += 1
        
        ## tee up next iteration; halt if no further connections can be made
        unpaired_edges = unpaired_updated
        n_iter += 1
        
        LOGGER.info(f'Paired up {n_paired_new} new edges after {n_iter} iteration(s); {len(unpaired_edges)}/{num_total_edges} edges remain unpaired')
        if n_paired_new == 0:
            LOGGER.info(f'No new edges paired, halting registration loop')
            break 
        # TODO: log exceedance of max number of loops?
        
    if any(unpaired_edges):
        raise EdgeEmbeddingError(f'Could not identify connection for every edge; try running registration procedure for >{n_iter_max} iterations, or check topology/Connectors')
        
    ## DEV: with the refactor to have all Child Connectors be external by default in Primitive...
    ## ...it's no longer necessary to compute which are external here (though we have enough info to do so, as shown)
    # collate remaining unpaired Connectors as external
    # external_connectors : dict[PrimitiveHandle, tuple[Connector]] = {
    #     owner_handle : tuple(chain.from_iterable(eq_classes.values()))
    #         for owner_handle, eq_classes in connector_equiv_classes.items()
    #             if eq_classes # skip over nodes whose equivalence classes have been exhausted
    # }
    # return paired_connectors, external_connectors
    
    return paired_connectors