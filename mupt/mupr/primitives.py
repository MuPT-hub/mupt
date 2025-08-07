'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import Any, Generator, Hashable, Optional, Union, get_args
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter

from scipy.spatial.transform import RigidTransform

from rdkit.Chem.rdchem import Atom, BondType
from rdkit.Chem.rdmolfiles import AtomFromSmiles, AtomFromSmarts

from .canonicalize import (
    Canonicalizable,
    canonical_graph_property,
    lex_order_multiset,
    lex_order_multiset_str,
)
from .ports import Port
from .topology import PolymerTopologyGraph
from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import apply_rigid_transformation_recursive


PrimitiveStructure = Union[PolymerTopologyGraph, Atom, int, None]
class BadPrimitiveStructure(TypeError):
    '''Exception raised when a Primitive is initialized with an invalid structure'''
    ...

class Primitive(ABC):
    '''Represents a fundamental (but not necessarily irreducible) building block of a polymer system in the abstract 
    Note that, by default ALL fields are optional; this is to reflect the fact that use-cases and levels of info provided may vary
    
    For example, one might object that functionality and number of atoms could be derived from the SMILES string and are therefore redundant;
    However, in the case where no chemistry is explicitly provided, it's still perfectly valid to define numbers of atoms present
    E.g. a coarse-grained sticker-and-spacer model
    
    As another example, a 0-functionality primitive is also totally legal (ex. as a complete small molecule in an admixture)
    But comes with the obvious caveat that, in a network, it cannot be incorporated into a larger component
    '''
    # initialization methods         
    def __init__(
            self,
            structure : PrimitiveStructure=None,
            ports : Optional[list[Port]]=None,
            shape : Optional[BoundedShape]=None,
            label : Optional[Hashable]=None,
            metadata : Optional[dict[Hashable, Any]]=None,
        ) -> None:
            # essential structural information
            self.structure = structure     # connection of internal parts (or lack thereof); used to find children in multiscale hierarchy - DEVNOTE: implicitly invokes structure.setter descriptor
            self.ports = ports or []       # a collection of sites representing bonds to other Primitives
            self.shape = shape             # a rigid shape which approximates and abstracts the behavoir of the primitive in space

            # additional descriptors
            self.label = label             # a handle for users to identify and distinguish Primitives by
            self.metadata = metadata or {} # literally any other information the user may want to bind to this Primitive  
    
    # DEVNOTE: have platform-specific initializers/exporters be imported from interfaces (a la OpenFF Interchange)   
       
    def copy(self) -> 'Primitive':
        '''Return a new Primitive with the same information as this one'''
        return Primitive(**self.__dict__) # TODO: deepcopy attributes dict?
    
    # properties derived from core components
    ## structural properties
    @property
    def structure(self) -> PrimitiveStructure:
        '''The internal chemical structure of this Primitive'''
        if not hasattr(self, '_structure'):
            raise AttributeError('Primitive structure not set during initialization')
        if not isinstance(self._structure, get_args(PrimitiveStructure)):
            raise TypeError(f'Primitive internal structure must be one of {get_args(PrimitiveStructure)}, not {type(self._structure)}')
        
        return self._structure

    @structure.setter
    @abstractmethod # DEVNOTE: <-- THIS is configured by Primitive subtypes
    def structure(self, new_structure : PrimitiveStructure) -> None:
        '''Safely set the internal chemical structure of this Primitive'''
        ...
    
    # @abstractmethod
    # def children(self, dfs : bool=True) -> Generator['Primitive', None, None]:
        # '''Generate all sub-Primitives contained within this Primitive'''
        # raise NotImplemented
    
    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        '''Whether the Primitive at hand is at the bottom of a structural hierarchy'''
        ...
        
    @property
    @abstractmethod
    def is_atomic(self) -> bool:
        '''Whether the Primitive at hand represents a single atom from the periodic table'''
        ...
    
    @property
    @abstractmethod
    def is_all_atom(self) -> bool:
        '''Whether ALL of a Primitive's structure can be traced to atoms occuring on the periodic table'''
        ...
        
    @property
    @abstractmethod
    def num_atoms(self) -> int:
        '''Number of atoms the Primitive and its internal structure collectively represent'''
        ...
    
    @abstractmethod    
    def canonical_form_structure(self) -> str:
        '''A canonical string representing this Primitive's structure'''
        ...
    
    ## Shape properties
    def canonical_form_shape(self) -> str: # DEVNOTE: for now, this doesn;t need to be abstract (just use type of Shapefor all kinds of Primitive)
        '''A canonical string representing this Primitive's shape'''
        return type(self.shape).__name__
    
    ## Port properties
    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        return len(self.ports)
    
    @property
    def bondtype_index(self) -> tuple[tuple[BondType, int], ...]:
        '''
        Canonical identifier of all unique BondTypes by count among the Ports associated to this Primitive
        Consists of all (integer bondtype, count) pairs, sorted lexicographically
        '''
        return lex_order_multiset(port.canonical_form() for port in self.ports)
    
    def canonical_form_ports(self, separator : str=':', joiner : str='-') -> str:
        '''A canonical string representing this Primitive's ports'''
        return lex_order_multiset_str(
            (port.canonical_form() for port in self.ports),
            element_repr=str, #lambda bt : BondType.values[int(bt)]
            separator=separator,
            joiner=joiner,
        )
    
    # identification and comparison methods
    def canonical_form(self) -> str: # NOTE: deliberately NOT a property to indicated computing this might be expensive
        '''A canonical representation of a Primitive's core parts; induces a natural equivalence relation on Primitives
        I.e. two Primitives having the same canonical form are to be considered interchangable within a polymer system
        '''
        return f'{self.canonical_form_structure()}({self.canonical_form_ports()})<{self.canonical_form_shape()}>'
    
    def canonical_form_peppered(self) -> str:
        '''
        Return a canonical string representation of the Primitive with peppered metadata
        Used to distinguish two otherwise-equivalent Primitives, e.g. as needed for graph embedding
        
        Named for the cryptography technique of augmenting a hash by some external, stored data
        (as described in https://en.wikipedia.org/wiki/Pepper_(cryptography))
        '''
        return f'{self.canonical_form()}-{self.label}' #{self.metadata}'

    def __str__(self) -> str: # NOTE: this is what NetworkX calls when auto-assigning labels (NOT __repr__!)
        return self.canonical_form_peppered()
    
    def __repr__(self):
        return f'{self.__class__.__name__}(structure_type={type(self.structure).__name__}, shape={type(self.shape).__name__}, functionality={self.functionality}, num_atoms={self.num_atoms}, label={self.label})'
    
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
        
    # geometric methods
    def apply_rigid_transformation(self, transform : RigidTransform) -> 'Primitive': # TODO: make this specifically act on shape, ports, and structure?
        '''Apply an isometric (i.e. rigid) transformation to all parts of a Primitive which support it'''
        return Primitive(**apply_rigid_transformation_recursive(self.__dict__, transform))
    
    

## particular kinds of Primitives suited for different types of molecular resolution
class StructuralPrimitive(Primitive):
    '''A Primitive which contains other Primitives, internal structure described the connectivity of member Primitives'''
    # structural properties
    @Primitive.structure.setter
    def structure(self, new_structure : PolymerTopologyGraph) -> None:
        if not isinstance(new_structure, PolymerTopologyGraph):
            raise BadPrimitiveStructure(f'Primitive structure must be a PolymerTopologyGraph, not {type(new_structure)}')
        self._structure = new_structure
    
    @property
    def is_leaf(self) -> bool:
        return False
    
    @property
    def is_atomic(self) -> bool:
        return False
    
    @property
    def is_all_atom(self) -> bool:
        return all(primitive.is_all_atom for primitive in self.structure) # utilize Primitive contract for is_all_atom property
        
    @property
    def num_atoms(self) -> int:
        _num_atoms : int = 0
        for subprimitive in self.structure:
            if not isinstance(subprimitive, Primitive): # TODO: move this to external TopologyGraph validator method
                raise TypeError(f'Primitive Topology improperly embedded; cannot determine number of atoms from non-Primitive {subprimitive}')
            _num_atoms += subprimitive.num_atoms
        return _num_atoms
        
    def canonical_form_structure(self) -> str:
        return canonical_graph_property(self.structure) # DEVNOTE: this is a placeholder; needs to be implemented
    
    # resolution shift methods - unique to StructuralPrimitives (i.e. Composites)
    def atomize(self, uniquify : bool=False) -> Generator['Primitive', None, None]:
        '''Decompose primitive into its unique, constituent single-atom primitives'''
        raise NotImplementedError
        
    def coagulate(self) -> 'Primitive':
        '''Combine all constituent primitives into a single, larger primitive'''
        raise NotImplementedError


class NeutralPrimitive(Primitive):
    '''A Primitive which has no defined structure, shape, or number of atoms
    Typically used to indicate that some essential chemical information has not been provided'''
    @Primitive.structure.setter
    def structure(self, new_structure : PrimitiveStructure) -> None:
        if new_structure is not None:
            LOGGER.warning(f'Neutral Primitive structure will be set to None, despite invalid provided value of {new_structure!r}')
        self._structure = None
    
    @property
    def is_leaf(self) -> bool:
        return True
        
    @property
    def is_atomic(self) -> bool:
        return False
    
    @property
    def is_all_atom(self) -> bool:
        return False
        
    @property
    def num_atoms(self) -> int:
        return 0
        
    def canonical_form_structure(self) -> str:
        return str(None) # NOTE: this could in principle be anything, as long as it can't be confused for an atom or graph hash
        
        
class AtomicPrimitive(Primitive):
    '''A Primitive which is a single atom from the periodic table, i.e. has no internal structure'''
    # DEVNOTE: opted not to constrain .shape from __init__ to allow for pointlike atoms, vdW sphere, rodlike atoms, or other Shapes
    @Primitive.structure.setter
    def structure(self, new_structure : Union[Atom, str]) -> None:
        if isinstance(new_structure, str): # attempt SMILES/SMARTS upconversion
            new_structure = AtomFromSmiles(new_structure)
            if new_structure is None: # fallback to more general SMARTS if string pattern is not recognized as SMILES
                new_structure = AtomFromSmarts(new_structure)
        
        if not isinstance(new_structure, Atom): # DEVNOTE: implicitly, catches None returned when str input is not a valid SMARTS either
            raise BadPrimitiveStructure(f'Primitive structure must be an Atom, not {type(new_structure)}')
        self._structure = new_structure
    
    @property
    def is_leaf(self) -> bool:
        return True
        
    @property
    def is_atomic(self) -> bool:
        return True
    
    @property
    def is_all_atom(self) -> bool:
        return True
        
    @property
    def num_atoms(self) -> int:
        return 1
        
    def canonical_form_structure(self) -> str:
        symbol : str = self.structure.GetSymbol()
        if self.structure.GetIsAromatic():
            symbol = symbol.lower() # for now, encode aromaticity thru case; see how well (or poorly) this generalizes later

        return symbol # TODO: make this more expressive to capture stereo, aromaticity etc


class CardinalPrimitive(Primitive):
    '''A Primitive which has a well-defined NUMBER of member atoms, but no explicit structure'''
    @Primitive.structure.setter
    def structure(self, new_structure : int) -> None:
        if not isinstance(new_structure, int):
            raise BadPrimitiveStructure(f'Primitive structure must be an integer, not {type(new_structure)}')
        if new_structure < 0:
            raise ValueError('Primitive cannot contain a negative number of atoms')
        
        self._structure = new_structure
    
    @property
    def is_leaf(self) -> bool:
        True
        
    @property
    def is_atomic(self) -> bool:
        False
    
    @property
    def is_all_atom(self) -> bool:
        # NOTE: while this Primitive is considered to contain atoms, the fact that they're not
        # made explicit means it's not useful to declare this Primitive capable of bearing atoms
        False 
        
    @property
    def num_atoms(self) -> int:
        return self.structure
        
    def canonical_form_structure(self) -> str:
        return f'#{self.structure}'


# Lexica for storing and indexing for "basis sets" of Primitives
@dataclass
class PrimitiveLexicon:
    '''Collection of primitives which form the basis of a'''
    primitives : dict[str, Primitive] = field(default_factory=dict) # primitives keyed by aliases
    
    # this will likely need **some** methods, but most of the sanitization ought to be the responsibility of the MID graph
    # this is more-or-less a glorified container for now
       