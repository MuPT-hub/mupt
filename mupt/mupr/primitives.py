'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import Any, Generator, Hashable, Iterable, Optional, TypeVar, Union
PrimitiveLabel = TypeVar('PrimitiveLabel', bound=Hashable)
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
from .embedding import register_topology
from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import RigidlyTransformable


class Primitive(NodeMixin, RigidlyTransformable):
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
    # Initializers
    def __init__(
        self,
        topology : TopologicalStructure=None,
        shape : Optional[BoundedShape]=None,
        element : Optional[Element]=None,
        connectors : list[Connector]=None,
        label : Optional[PrimitiveLabel]=None,
        metadata : dict[Hashable, Any]=None,
    ) -> None:
        # essential components
        self.shape = shape
        self.element = element
        self.connectors = connectors or list()
        ## NOTE: setting of topology deliberately placed at end so validation is preformed based on values of other core attrs
        self.topology = topology or TopologicalStructure() # NOTE: the empty-set topology is valid EVEN if there are no children 
        
        # additional descriptors
        self.label = label
        self.metadata = metadata or dict()
        
        # non-init attrs - placed here for bookkeeping
        paired_connectors: dict[tuple[Hashable, Hashable], tuple[Connector, Connector]] = dict()
        external_connectors: dict[Hashable, tuple[Connector]] = dict()


    # Fulfilling inheritance contracts
    def _copy_untransformed(self) -> 'Primitive':
        '''Return a new Primitive with the same information as this one'''
        return self.__class__(
            topology=TopologicalStructure(self.topology),
            shape=(None if self.shape is None else self.shape.copy()),
            element=self.element,
            connectors=[conn.copy() for conn in self.connectors],
            label=self.label,
            metadata={key : value for key, value in self.metadata.items()},
        ) # TODO: deepcopy attributes dict?
    # TODO: copy hierarchy of children
    
    def _rigidly_transform(self, transform : RigidTransform) -> None: 
        '''Apply a rigid transformation to all parts of a Primitive which support it'''
        if isinstance(self.shape, BoundedShape):
            self.shape.rigidly_transform(transform)
        
        for connector in self.connectors:
            connector.rigidly_transform(transform)
            
        # propogate transformation down recursively
        for subprimitive in self.children: 
            subprimitive.rigidly_transform(transform)

    
    # Chemical atom and bond properties
    @property
    def element(self) -> Optional[Element]:
        '''The chemical element associated with this Primitive, if it represents an atom'''
        if not hasattr(self, '_element'):
            return None
        return self._element
    
    @element.setter
    def element(self, new_element: Optional[Element]) -> None:
        if (new_element is not None):
            if self.children:
                raise AttributeError('Primitive with non-trivial internal structure cannot be made atomic (i.e. cannot have "element" assigned)')
            if not isinstance(new_element, Element):
                raise TypeError(f'Invalid element type {type(new_element)}')
        self._element = new_element
    
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
    def bondtype_index(self) -> tuple[tuple[Any, int], ...]:
        '''
        Canonical identifier of all unique BondTypes by count among the Connectors associated to this Primitive
        Consists of all (integer bondtype, count) pairs, sorted lexicographically
        '''
        return lex_order_multiset(connector.canonical_form() for connector in self.connectors)
    
    
    # Networking
    ## Child Primitives
    @property
    def n_children(self) -> int:
        '''Number of sub-Primitives this Primitive contains'''
        return len(self.children)

    def children_are_uniquely_labelled(self) -> bool:
        '''Check if that no pair of child Primitives are assigned the same label'''
        if not self.children:
            return True
        labels = [child.label for child in self.children]
        
        return len(labels) == len(set(labels))

    def child_label_classes(self) -> dict[PrimitiveLabel, tuple['Primitive']]:
        '''Return equivalence classes of child Primitives by their assigned labels''' # DEVNOTE: transition to canonical forms, eventually?
        _child_map = defaultdict(list)
        for subprim in self.children:
            _child_map[subprim.label].append(subprim)
            
        return { # DEV: recast as equivalence classes by label?
            label : tuple(subprims)
                for label, subprims in _child_map.items()
        }
        
    @property
    def children_by_label(self) -> dict[PrimitiveLabel, 'Primitive']:
        '''Get child Primitive by its (presumed-unique) label'''
        if not self.children_are_uniquely_labelled():
            raise ValueError(f'Injective mapping of labels onto child Primitives impossible, since labels amongst chilren are not unique')
        
        return {
            label : subprims[0]
                for label, subprims in self.child_label_classes().items()
        }
        
    def attach_child(
            self,
            subprimitive : 'Primitive',
            neighbor_labels : Optional[Iterable[PrimitiveLabel]]=None,
        ) -> None:
        '''Add another Primitive as a child of this one, updating topology in accordance'''
        # G.add_edges_from

        ## TODO: deduce compatible Connectors for neighbors provided?
        raise NotImplementedError

    def detach_child(
            self,
            subprimitive : Union['Primitive', PrimitiveLabel],
        ) -> None:
        '''Remove a child Primitive from this one, updating topology and Connectors'''
        # G.remove_node()
        raise NotImplementedError
    
    # DEV: also include attach_/detach_parent?

    ## Connections
    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        if not hasattr(self, 'connectors'):
            self.connectors = list() # needed, for example, when checking functionality during init
        return len(self.connectors)
    
    ## Topology
    @property
    def topology(self) -> Optional[TopologicalStructure]:
        '''The connectivity of the immediate children of this Primitive, if one is defined'''
        if not hasattr(self, '_topology'):
            return None
        return self._topology

    @topology.setter
    def topology(self, new_topology: Optional[TopologicalStructure]) -> None:
        # TODO: initialize discrete topology with number of nodes equal to number of children
        if not isinstance(new_topology, TopologicalStructure):
            raise TypeError(f'Invalid topology type {type(new_topology)}')
        
        if not self.topology_is_compatible(new_topology):
            raise ValueError('Provided topology is incompatible with the sub-Primitives associated to this Primitive')

        self._topology = new_topology

    def _check_children_bijective_to_topology(self, topology: TopologicalStructure) -> None:
        '''
        Check whether a 1:1 correspondence can exist between all child Primitives and all elements of the imposed incidence topology
        Raises descriptive Exception is no correspondence can exist, or else returns silently
        '''
        # NOTE: logic here handles leaf and non-leaf cases uniformly - no extra branching required
        if topology.number_of_nodes() != self.n_children:
            raise ValueError(f'Cannot bijectively map {self.n_children} child Primitives onto {topology.number_of_nodes()}-element topology')
        
        child_map : dict[PrimitiveLabel, tuple['Primitive']] = self.children_by_label # implicitly checks uniqueness of labels
        child_labels : set[PrimitiveLabel] = set(child_map.keys())
        node_labels  : set[PrimitiveLabel] = set(topology.nodes)
        if node_labels != child_labels:
            raise KeyError(
                f'Underlying set of topology does not correspond to labels on child Primitives; {len(node_labels - child_labels)} elements'\
                f' have no associated children, and {len(child_labels - node_labels)} children are unrepresented in the topology'
            )

    def _check_functionalities_compatible_with_topology(self, topology: TopologicalStructure) -> bool:
        '''Check whether the functionalities of all child Primitives are compatible with the imposed Topology'''
        if self.is_leaf:
            if self.is_atom:
                ## TODO: include valency checks based on atomic number and formal charge
                ...
            else:
                ## for non-atom leaves, not really clear as of yet whether there even exist and universally-required conditions for functionality
                ...
            return # for now, exit early w/o Exception for leaf cases

        ## 1) check local connectivity doesn't preclude embedding
        self._check_children_bijective_to_topology(topology) # ensure we know which children correspond to which nodes first
        n_external_connectors : int = 0
        for subprimitive in self.children:
            min_degree : int = topology.degree[subprimitive.label]
            n_excess = subprimitive.functionality - min_degree
            if n_excess < 0:
                raise ValueError(f'Cannot embed {subprimitive.functionality}-functional Primitive "{subprimitive.label}" into {min_degree}-degree node')
            n_external_connectors += n_excess

        ## 2) check global excess Connectors matches number at next level in hierarchy
        if n_external_connectors != self.functionality:
            raise ValueError(f'Cannot bijectively map {n_external_connectors} external connectors from children onto connections of {self.functionality}-functional Primitive')

    def topology_is_compatible(self, topology: TopologicalStructure) -> bool:
        '''Verify the topology induced on this Primitive's children is valid'''
        # Perform simpler counting checks first, to fail fast in case an embedding obviously can't exist
        # NOTE: node bijection checks already part of functionality check
        self._check_functionalities_compatible_with_topology(topology)

        # check to see that Connectors can be paired up 1:1 along edges
        ## TODO: propagate error message and return False
        paired_connectors, external_connectors = register_topology( # will raise exception is registration is not possible
            labelled_connectors={
                subprim.label : subprim.connectors
                    for subprim in self.children
            },
            topology=topology
        )
        self.paired_connectors = paired_connectors
        self.external_connectors = external_connectors
        ## TODO: identify WHICH Connectors at this level match to which external connectors found among children

        return True
    
    
    # Geometry (info about Shape and transformations)
    @property
    def shape(self) -> Optional[BoundedShape]:
        '''The external shape of this Primitive'''
        if not hasattr(self, '_shape'):
            return None
        return self._shape
    
    @shape.setter
    def shape(self, new_shape : Optional[BoundedShape]) -> None:
        '''Set the external shape of this Primitive'''
        if (new_shape is not None) and (not isinstance(new_shape, BoundedShape)):
            raise TypeError(f'Primitive shape must be either NoneType or BoundedShape instance, not object of type {type(new_shape.__name__)}')

        if not isinstance(self.shape, BoundedShape): # DEV: no typo here; deliberate call to getter to handle case when _shape (not "shape"!) is unset
            self._shape = new_shape
        else:
            new_shape_clone = new_shape.copy() # NOTE: make copy to avoid mutating original (per Principle of Least Astonishment)
            new_shape_clone.cumulative_transformation = self.shape.cumulative_transformation # transfer translation history BEFORE overwriting
            self._shape = new_shape_clone
            
    # Representation methods
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
    
    def coincident_with(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are coincident (i.e. all spatial parts are either equally unassigned or occupy the same space)'''
        raise NotImplementedError
    
    def equivalent_to(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are equivalent (i.e. have interchangeable part which are not necessarily in the same place in space)'''
        raise NotImplementedError

    ## display methods
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
