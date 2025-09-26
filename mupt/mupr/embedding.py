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
)
T = TypeVar('T')

from itertools import (
    chain,
    product as cartesian,
)
from networkx import Graph
from networkx.utils import arbitrary_element

# DEV: this module CANNOT import Primitive if circular imports are to be avoided
from .connection import Connector
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

def register_connectors_to_topology(
    labelled_connectors : Mapping[Hashable, Iterable[Connector]],
    topology : TopologicalStructure,
    n_iter_max : int=3,
) -> tuple[
        dict[tuple[Hashable, Hashable], dict[Hashable, Connector]],
        dict[Hashable, tuple[Connector]]
    ]:
    """
    Deduce if a collection of Connectors associated to each node in a topology
    can be identified with the edges in that topology, such that each pair of Connectors is bondable
    
    Returns a first mapping of pairs of node labels (one pair for each edge)
    to a mapping from node labels to the Connector associated to that edge,
    and a second mapping of node labels to remaining external Connectors, if any remain unpaired
    
    If pairing is impossible, will raise Exception instead
    """
    if not isinstance(topology, Graph):
        raise TypeError(f'Topology must be a Graph instance, not one of type {type(topology)}')
    
    if not set(topology.nodes).issubset(set(labelled_connectors.keys())): 
        # weaker requirement of containing (rathe than being equal) to vertex set suffices
        # DEV: replace labels w/ handle, eventually; presumes a mapping of Primitives onto the nodes exists
        raise NodeEmbeddingError('Connector collection labels do not match topology node labels')

    # Initialized containers for tracking pairing progress
    paired_connectors : dict[tuple[Hashable, Hashable], tuple[Connector, Connector]] = {
        edge : tuple()
            for edge in topology.edges
    }
    connector_equiv_classes : dict[Hashable, dict[int, list[Connector]]] = {
        owner_label : mapped_equivalence_classes(connectors, Connector.fungible_with)
            for owner_label, connectors in labelled_connectors.items()
    }

    # iteratively pair connectors along edges
    n_iter : int = 0
    while n_iter < n_iter_max:
        n_paired_new : int = 0
        for edge_labels in topology.edges:
            ## skip check for edges already assigned a pair of Connectors
            if paired_connectors[edge_labels]:
                LOGGER.debug(f'Skipping already-paired edge designated "{edge_labels}"')
                continue 
            
            ## attempt to identify if there is a UNIQUE pair of bondable classes of Connectors along the edge
            pair_choice_ambiguous : bool = False
            compatible_class_labels : Optional[tuple[Connector, Connector]] = None
            for (class_label1, eq_class1), (class_label2, eq_class2) in cartesian(
                    connector_equiv_classes[edge_labels[0]].items(),
                    connector_equiv_classes[edge_labels[1]].items(),
                ):
                if not Connector.bondable_with( # DEV: opted for this callstyle to highlight symmetry of compariso
                        arbitrary_element(eq_class1),
                        arbitrary_element(eq_class2),
                    ): 
                    continue # skip over incompatible Connector classes
                
                if compatible_class_labels is None:
                    compatible_class_labels = (class_label1, class_label2)
                else:
                    pair_choice_ambiguous = True
                    break # further search can't disambiguate choice, stop early to save computation
                
            if pair_choice_ambiguous:
                LOGGER.debug(f'Choice of Connector pair ambiguous for edge {edge_labels}, skipping')
                continue
            elif (compatible_class_labels is None):
                raise EdgeEmbeddingError(f'No compatible Connector pairs found for edge {edge_labels}')

            ## if unambiguous pairing is present, draw representatives of respective compatible classes and bind them
            chosen_representatives : dict[Hashable, Connector] = dict()
            for (class_label, node_label) in zip(compatible_class_labels, edge_labels):
                equiv_class = connector_equiv_classes[node_label][class_label]
                chosen_representatives[node_label] = equiv_class.pop(0) # DEV: index here shouldn't matter, but will standardized to match arbitrary element selection
                
                ### remove bin from equivalence class if empty after drawing
                if len(equiv_class) == 0: 
                    _ = connector_equiv_classes[node_label].pop(class_label)
            paired_connectors[edge_labels] = chosen_representatives
            n_paired_new += 1
        
        ## tee up next iteration; halt if no further connections can be made
        n_iter += 1
        LOGGER.debug(f'Paired up {n_paired_new} new edges after {n_iter} iteration(s)')
        if n_paired_new == 0:
            LOGGER.debug(f'No new edges paired, halting registration loop')
            break 
        # TODO: log exceedance of max number of loops?
        
    if not all(paired_connectors.values()):
        raise EdgeEmbeddingError(f'No complete pairing of Connectors found; try running registration procedure for >{n_iter_max} iterations, or check topology/connectors')
        
    # collate remaining unpaired Connectors as external
    external_connectors : dict[Hashable, tuple[Connector]] = {
        owner_label : tuple(chain.from_iterable(eq_classes.values()))
            for owner_label, eq_classes in connector_equiv_classes.items()
                if eq_classes # skip over nodes whose equivalence classes have been exhausted
    }
    
    return paired_connectors, external_connectors