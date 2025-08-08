'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import Any, Generator, Hashable, Optional
from dataclasses import dataclass, field

from scipy.spatial.transform import RigidTransform

from .canonicalize import (
    Canonicalizable,
    lex_order_multiset,
    lex_order_multiset_str,
)
from .connection import Connector
from .structure import Structure
from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import apply_rigid_transformation_recursive


class BadPrimitiveStructure(TypeError):
    '''Exception raised when a Primitive is initialized with an invalid structure'''
    ...

@dataclass
class Primitive:
    '''Represents a fundamental (but not necessarily irreducible) building block of a polymer system in the abstract 
    Note that, by default ALL fields are optional; this is to reflect the fact that use-cases and levels of info provided may vary
    
    For example, one might object that functionality and number of atoms could be derived from the SMILES string and are therefore redundant;
    However, in the case where no chemistry is explicitly provided, it's still perfectly valid to define numbers of atoms present
    E.g. a coarse-grained sticker-and-spacer model
    
    As another example, a 0-functionality primitive is also totally legal (ex. as a complete small molecule in an admixture)
    But comes with the obvious caveat that, in a network, it cannot be incorporated into a larger component
    '''
    # essential components
    structure  : Optional[Structure] = field(default=None),    # connection of internal parts (or lack thereof); used to find children in multiscale hierarchy - DEVNOTE: implicitly invokes structure.setter descriptor
    shape      : Optional[BoundedShape] = field(default=None), # a rigid shape which approximates and abstracts the behavoir of the primitive in space
    connectors : list[Connector] = field(default_factory=list),     # a collection of sites representing bonds to other Primitives
    # additional descriptors
    label    : Optional[Hashable] = field(default=None),         # a handle for users to identify and distinguish Primitives by
    metadata : dict[Hashable, Any] = field(default_factory=dict) # literally any other information the user may want to bind to this Primitive  

    # initializers
    # DEVNOTE: have platform-specific initializers/exporters be imported from interfaces (a la OpenFF Interchange)   
    def copy(self) -> 'Primitive':
        '''Return a new Primitive with the same information as this one'''
        return Primitive(**self.__dict__) # TODO: deepcopy attributes dict?
    
    # connection properties
    @property
    def num_atoms(self) -> int:
        '''Number of atoms the Primitive and its internal structure collectively represent'''
        return self.structure.num_atoms
    
    def components(self, dfs : bool=True) -> Generator['Primitive', None, None]:
        '''Generate all sub-Primitives contained within this Primitive'''
        yield from self.structure.components()
        
    @property
    def is_leaf(self) -> bool:
        '''Whether the Primitive at hand is at the bottom of a structural hierarchy'''
        return not self.structure.is_composite
        
    # Connector properties
    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        return len(self.connectors)
    
    @property
    def bondtype_index(self) -> tuple[tuple[Any, int], ...]:
        '''
        Canonical identifier of all unique BondTypes by count among the Connectors associated to this Primitive
        Consists of all (integer bondtype, count) pairs, sorted lexicographically
        '''
        return lex_order_multiset(connector.canonical_form() for connector in self.connectors)

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
        return f'{self.structure.canonical_form()}({self.canonical_form_connectors()})<{self.canonical_form_shape()}>'

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
    
    def equivalent_to(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are equivalent (i.e. interchangeable, but not necessarily identical)'''
        raise NotImplementedError
    
    def coincident_with(self, other : 'Primitive') -> bool:
        '''Check whether two Primitives are coincident (i.e. occupy the same space)'''
        raise NotImplementedError

    # display methods
    def __str__(self) -> str: # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
        return self.canonical_form_peppered()
    
    def __repr__(self):
        repr_attr_strs : dict[str, str] = {
            'shape': self.canonical_form_shape(),
            'functionality': str(self.functionality),
            'structure_type': type(self.structure).__name__,
            'label': self.label
        }
        attr_str = ', '.join(
            f'{attr}={value_str}'
                for (attr, value_str) in repr_attr_strs.items()
        )
        
        return f'{self.__class__.__name__}({attr_str})'
        
    # geometric methods
    def apply_rigid_transformation(self, transform : RigidTransform) -> 'Primitive': # TODO: make this specifically act on Shape, Connectors, and Structure?
        '''Apply an isometric (i.e. rigid) transformation to all parts of a Primitive which support it'''
        return Primitive(**apply_rigid_transformation_recursive(self.__dict__, transform))
    