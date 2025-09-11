'''Utilities for inserting Primitives into pre-built graphs to create polymer topology graphs'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

# DEVNOTE: this is not a submodule under topology to avoid circular imports
# and to shelter MID Graphs from needing to know about HOW they're embedded

import logging
LOGGER = logging.getLogger(__name__)

from typing import Callable, Hashable, Iterable, Mapping, TypeVar
T = TypeVar('T')

from itertools import (
    chain,
    product as cartesian,
)
from networkx import Graph
from networkx.utils import arbitrary_element

from .primitives import Primitive, PrimitiveLabel
from .connection import Connector
from .topology import TopologicalStructure


def equivalence_classes(objects : Iterable[T], relation : Callable[[T, T], bool]) -> Iterable[list[T]]:
    """
    Partition a collection of objects into equivalence classes by
    an equivalence relation defined on pairs of those objects
    """
    equiv_classes : list[list[T]] = []
    for obj in objects:
        for equiv_class in equiv_classes:
            if relation(obj, arbitrary_element(equiv_class)):
                equiv_class.append(obj)
                break
        else:
            equiv_classes.append([obj])
    
    return equiv_classes

def register_topology(
    labelled_primitives : Mapping[PrimitiveLabel, Primitive],
    topology : TopologicalStructure,
    n_iter_max : int=3,
) -> tuple[
        dict[tuple[PrimitiveLabel, PrimitiveLabel], tuple[Connector, Connector]],
        dict[PrimitiveLabel, tuple[Connector]]
    ]:
    """
    Deduce if the Connectors within each of a collection of Primitives can be identified with
    the edges in a topology on those Primitives, such that each pair of Connectors is bondable
    
    Returns mapping of pairs of Primitives (along edges) to their associated pairs of Connectors,
    and a mapping of Primitives to remaining external Connectors, if any remain unpaired
    
    If pairing is impossible, will raise Exception instead
    """
    if not isinstance(topology, Graph):
        raise TypeError(f'Topology must be a Graph instance, not one of type {type(topology)}')
    
    if not set(topology.nodes).issubset(set(labelled_primitives.keys())): 
        # set of Primitives is allowed to be strictly larger than the topology on it, mapped labels implicitly enforces uniqueness of Primitive labels 
        # DEV: replace labels w/ handle, eventually; presumes a mapping of Primitives onto the nodes exists
        raise ValueError('Primitive labels do not match topology node labels')

    # Initialized containers for tracking pairing progress
    paired_connectors : dict[tuple[PrimitiveLabel, PrimitiveLabel], tuple[Connector, Connector]] = {
        edge : tuple()
            for edge in topology.edges
    }
    connector_equiv_classes : dict[PrimitiveLabel, dict[int, list[Connector]]] = {
        label : {
            i : equiv_class # DEV: opt for index as label for now; eventually want label to be related to Connectors within equivalence class
                for i, equiv_class in enumerate(equivalence_classes(primitive.connectors, Connector.fungible_with))
        }
            for label, primitive in labelled_primitives.items()
    }

    # iteratively pair connectors along edges
    n_iter : int = 0
    while n_iter < n_iter_max:
        n_paired_new : int = 0
        for edge_label in topology.edges:
            if paired_connectors[edge_label]:
                LOGGER.debug(f'Skipping already-paired edge designated "{edge_label}"')
                continue # skip check for edges already assigned a pair of Connectors
            
            connector_equiv_classes1 = connector_equiv_classes[edge_label[0]]
            connector_equiv_classes2 = connector_equiv_classes[edge_label[1]]

            compatible_conns : set[tuple[Connector, Connector]] = set()
            for (class_label1, eq_class1), (class_label2, eq_class2) in cartesian(connector_equiv_classes1.items(), connector_equiv_classes2.items()):
                if arbitrary_element(eq_class1).bondable_with(arbitrary_element(eq_class2)):
                    compatible_conns.add( (class_label1, class_label2) )
            
            if len(compatible_conns) == 0: # declare failure for unbonded edges with no compatible connections
                raise ValueError(f'No compatible connector pairs found for edge {edge_label}')
            elif len(compatible_conns) > 1:
                LOGGER.debug(f'Choice of Connector pair ambiguous for edge {edge_label}, skipping')
                continue # TODO: move this into cartesian loop to save on checks
            else:
                chosen_representatives : list[Connector] = []
                class_labels = arbitrary_element(compatible_conns) # extract indices of lone compatible pair from set
                for (class_label, node_label) in zip(class_labels, edge_label):
                    equiv_class = connector_equiv_classes[node_label][class_label]
                    chosen_representatives.append(equiv_class.pop(0)) # DEV: index here shouldn't matter, but will standardized to match arbitrary element selection
                    if len(equiv_class) == 0: # remove bin for equivalence class if empty after drawing
                        _ = connector_equiv_classes[node_label].pop(class_label)

                paired_connectors[edge_label] = tuple(chosen_representatives)
                n_paired_new += 1
                
        n_iter += 1
        LOGGER.info(f'Paired up {n_paired_new} new edges after {n_iter} iteration(s)')
        if n_paired_new == 0:
            LOGGER.info(f'No new edges paired, halting registration loop')
            break # halt after no further connections can be made
        # TODO: log exceedance of max number of loops?
        
    if not all(paired_connectors.values()):
        raise ValueError(f'No complete pairing of Connectors found; try running registration procedure for >{n_iter_max} iterations, or check topology/connectors')
        
    external_connectors : dict[PrimitiveLabel, tuple[Connector]] = {
        label : tuple(chain.from_iterable(eq_classes.values()))
            for label, eq_classes in connector_equiv_classes.items()
                if eq_classes # skip over nodes whose equivalence classes have been exhausted
    }
    
    return paired_connectors, external_connectors