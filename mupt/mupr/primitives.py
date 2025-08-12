'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import Any, Generator, Hashable, Optional
from collections import defaultdict

from anytree.node import NodeMixin
from anytree.search import findall_by_attr

from periodictable.core import Element
from scipy.spatial.transform import RigidTransform

from .canonicalize import (
    Canonicalizable,
    lex_order_multiset,
    lex_order_multiset_str,
)
from .connection import Connector
from .topology import TopologicalStructure
from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import apply_rigid_transformation_recursive


class Primitive(NodeMixin):
    '''Represents a fundamental (but not necessarily irreducible) building block of a polymer system in the abstract 
    Note that, by default ALL fields are optional; this is to reflect the fact that use-cases and levels of info provided may vary
    
    For example, one might object that functionality and number of atoms could be derived from the SMILES string and are therefore redundant;
    However, in the case where no chemistry is explicitly provided, it's still perfectly valid to define numbers of atoms present
    E.g. a coarse-grained sticker-and-spacer model
    
    As another example, a 0-functionality primitive is also totally legal (ex. as a complete small molecule in an admixture)
    But comes with the obvious caveat that, in a network, it cannot be incorporated into a larger component
    
    Parameters
    ----------
    topology : TopologicalStructure
        connection of internal parts (or lack thereof); used to find children in multiscale hierarchy
    shape : Optional[BoundedShape]
        A rigid shape which approximates and abstracts the behavior of the primitive in space
    element : Optional[Element]
        The chemical element associated with this Primitive, IFF the Primitive represents an atom
    connectors : list[Connector]
        A collection of sites representing bonds to other Primitives

    label : Optional[Hashable]
        A handle for users to identify and distinguish Primitives by
    metadata : dict[Hashable, Any]
        Literally any other information the user may want to bind to this Primitive
    '''
    # initializers
    def __init__(
        self,
        topology : TopologicalStructure=None,
        shape : Optional[BoundedShape]=None,
        element : Optional[Element]=None,
        connectors : list[Connector]=None,
        label : Optional[Hashable]=None,
        metadata : dict[Hashable, Any]=None,
    ) -> None:
        # essential components
        self.topology = topology or TopologicalStructure() # NOTE: the empty-set topology is valid EVEN if there are no children
        self.shape = shape
        self.element = element # DEVNOTE: implicitly invokes topology.setter descriptor
        self.connectors = connectors or list()
        # additional descriptors
        self.label = label
        self.metadata = metadata or dict()
        
    # DEVNOTE: have platform-specific initializers/exporters be imported from interfaces (a la OpenFF Interchange)   
    def copy(self) -> 'Primitive':
        '''Return a new Primitive with the same information as this one'''
        return self.__class__(
            topology=TopologicalStructure(self.topology),
            shape=self.shape, # TODO: deepcopy this
            element=self.element,
            connectors=[conn.copy() for conn in self.connectors],
            label=self.label,
            metadata={key : value for key, value in self.metadata.items()},
        ) # TODO: deepcopy attributes dict?
    
    # descriptors for core attributes
    @property
    def element(self) -> Optional[Element]:
        '''The chemical element associated with this Primitive, if it represents an atom'''
        return self._element
    
    @element.setter
    def element(self, new_element: Optional[Element]) -> None:
        if new_element is not None and not isinstance(new_element, Element):
            raise TypeError(f'Invalid element type {type(new_element)}')
        self._element = new_element
      
    ## validating chosen topology     
    @property
    def topology(self) -> Optional[TopologicalStructure]:
        '''The connectivity of the immediate children of this Primitive, if one is defined'''
        return self._topology

    @topology.setter
    def topology(self, new_topology: Optional[TopologicalStructure]) -> None:
        # TODO: initialize discrete topology with number of nodes equal to number of children
        if not isinstance(new_topology, TopologicalStructure):
            raise TypeError(f'Invalid topology type {type(new_topology)}')
        
        if not self.topology_is_valid(new_topology):
            raise ValueError('Provided topology is incompatible with the set of child Primitives')

        self._topology = new_topology
    
    # connection properties
    @property
    def is_atom(self) -> bool:
        '''Whether the Primitive at hand represents a single atom'''
        return self.element is not None

    @property
    def num_atoms(self) -> int:
        '''Number of atoms the Primitive and its internal topology collectively represent'''
        return len(findall_by_attr(self, value=True, name='is_atom'))

    @property
    def is_atomizable(self) -> bool:
        '''Whether the Primitive represents an all-atom system'''
        return (self.is_atom and not self.children) or all(subprim.is_atomizable for subprim in self.children)
    
    @property
    def n_children(self) -> int:
        '''Number of sub-Primitives this Primitive contains'''
        return len(self.children)

    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        if not hasattr(self, 'connectors'):
            self.connectors = list() # needed, for example, when checking functionality during init
        return len(self.connectors)
    
    @property
    def bondtype_index(self) -> tuple[tuple[Any, int], ...]:
        '''
        Canonical identifier of all unique BondTypes by count among the Connectors associated to this Primitive
        Consists of all (integer bondtype, count) pairs, sorted lexicographically
        '''
        return lex_order_multiset(connector.canonical_form() for connector in self.connectors)

    # embedding and topology consistency checks
    def children_uniquely_labelled(self) -> bool:
        '''Check if that no pair of child Primitives are assigned the same label'''
        if not self.children:
            return True
        labels = [child.label for child in self.children]
        
        return len(labels) == len(set(labels))
    
    def child_label_classes(self) -> dict[Hashable, tuple['Primitive']]:
        '''Return equivalence classes of child Primitives by their assigned labels''' # DEVNOTE: transition to canonical forms, eventually?
        _child_map = defaultdict(list)
        for subprim in self.children:
            _child_map[subprim.label].append(subprim)
            
        return {
            label : tuple(subprims)
                for label, subprims in _child_map.items()
        }
        
    @property
    def children_by_label(self) -> dict[Hashable, 'Primitive']:
        '''Get child Primitive by its (presumed-unique) label'''
        if not self.children_uniquely_labelled():
            raise ValueError(f'Injective mapping of labels onto child Primitives impossible, since labels amongst chilren are not unique')
        
        return {label : subprims[0] for label, subprims in self.child_label_classes().items()}

    def topology_is_valid(self, topology : TopologicalStructure) -> bool: # TODO: add a version of this with descriptive errors
        '''Verify the topology induced on this Primitive's children is valid''' # DEVNOTE: make this staticmethod/classmethod?
        # Perform simpler checks first, to fail fast in case an embedding obviously can't exist
        ## check bijection between nodes and children
        if topology.number_of_nodes() != self.n_children: 
            LOGGER.error(f'Cannot bijectively map {self.n_children} child Primitives onto {topology.number_of_nodes()}-element topology')
            return False
        
        ## check balance over incident pair and Ports (external AND internal)
        num_connectors_internal : int = sum(subprim.functionality for subprim in self.children) - self.functionality # subtract off contribution from external connectors
        if num_connectors_internal != 2*topology.number_of_edges():
            LOGGER.error(f'Mismatch between {num_connectors_internal} internal connectors and 2*{topology.number_of_edges()} connectors required by topology')
            return False
        
        # perform more detailed checks on the connectivity of the topology
        # TODO: more complex check to see that children can be mapped 1:1 onto Nodes
        # TODO: more complex check to see that Ports can be paired up 1:1 along edges

        return True
    
    def validate_topology(self) -> bool:
        '''Check that the currently-set topology is compatible with the currently-defined children of this Primitive'''
        return self.topology_is_valid(self.topology)
    
    def embed_topology(self) -> None:
        '''Map sub-Primitives onto nodes in internal topology'''
        raise NotImplementedError

    # comparison methods
    ## canonical forms for core components
    def canonical_form_connectors(self, separator : str=':', joiner : str='-') -> str:
        '''A canonical string representing this Primitive's Connectors'''
        return lex_order_multiset_str(
            (connector.canonical_form() for connector in self.connectors),
            element_repr=str, #lambda bt : BondType.values[int(bt)]
            separator=separator,
            joiner=joiner,
        )
    
    def canonical_form_shape(self) -> str: # DEVNOTE: for now, this doesn't need to be abstract (just use type of Shapefor all kinds of Primitive)
        '''A canonical string representing this Primitive's shape'''
        return type(self.shape).__name__ # TODO: move this into .shape - should be responsibility of individual Shape subclasses
    
    def canonical_form(self) -> str: # NOTE: deliberately NOT a property to indicated computing this might be expensive
        '''A canonical representation of a Primitive's core parts; induces a natural equivalence relation on Primitives
        I.e. two Primitives having the same canonical form are to be considered interchangable within a polymer system
        '''
        elem_form : str = self.element.symbol if self.element is not None else str(None) # TODO: move this to external function, eventually
        return f'{elem_form}({self.canonical_form_connectors()})[{self.canonical_form_shape()}]<{self.topology.canonical_form()}>'

    def canonical_form_peppered(self) -> str:
        '''
        Return a canonical string representation of the Primitive with peppered metadata
        Used to distinguish two otherwise-equivalent Primitives, e.g. as needed for graph embedding
        
        Named for the cryptography technique of augmenting a hash by some external, stored data
        (as described in https://en.wikipedia.org/wiki/Pepper_(cryptography))
        '''
        return f'{self.canonical_form()}-{self.label}' #{self.metadata}'

    ## comparison methods
    def __hash__(self): 
        '''Hash used to compare Primitives for identity (NOT equivalence)'''
        # return hash(self.canonical_form())
        return hash(self.canonical_form_peppered())
    
    def __eq__(self, other : object) -> bool:
        # DEVNOTE: in order to use equivalent-but-not-identical Primitives as nodes in nx.Graph, __eq__ CANNOT evaluate similarity by hashes
        # DEVNOTE: hashing needs to be stricter than equality, i.e. two Primitives may be distinguishable by hash, but nevertheless equivalent
        '''Check whether two primitives are equivalent (but not necessarily identical)'''
        if not isinstance(other, Primitive):
            raise TypeError(f'Cannot compare Primitive to {type(other)}')

        return self.canonical_form() == other.canonical_form() # NOTE: ignore labels, simply check equivalency up to canonical forms
    
    def congruent(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are congruent (i.e. have interchangeable part which are not necessarily in the same place in space)'''
        raise NotImplementedError
    
    def coincident_with(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are coincident (i.e. all spatial parts are either equally unassigned or occupy the same space)'''
        raise NotImplementedError

    # display methods
    def __str__(self) -> str: # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
        return self.canonical_form_peppered()
    
    def __repr__(self):
        repr_attr_strs : dict[str, str] = {
            'shape': self.canonical_form_shape(),
            'functionality': str(self.functionality),
            'topology_type': repr(self.topology),
            'label': self.label
        }
        attr_str = ', '.join(
            f'{attr}={value_str}'
                for (attr, value_str) in repr_attr_strs.items()
        )
        
        return f'{self.__class__.__name__}({attr_str})'
        
    # geometric methods
    def apply_rigid_transformation(self, transform : RigidTransform) -> 'Primitive': 
        '''Apply an isometric (i.e. rigid) transformation to all parts of a Primitive which support it'''
        # TODO: make this act specifically on internal components, rather than generic recursive application
        return Primitive(**apply_rigid_transformation_recursive(self.__dict__, transform))
    