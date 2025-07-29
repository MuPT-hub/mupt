'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Generator, Hashable, Optional, Union, get_args
from dataclasses import dataclass, field
from collections import Counter

from scipy.spatial.transform import RigidTransform

from rdkit.Chem.rdchem import (
    Atom,
    Bond,
    Mol,
    BondType,
    # DEVNOTE: not yet sure what the best way to represent stereochemistry is
    StereoInfo,
    StereoType,
    StereoDescriptor,
    ChiralType,
)

from .ports import Port
from .topology import PolymerTopologyGraph

from ..geometry.shapes import BoundedShape
from ..geometry.transforms.rigid import apply_rigid_transformation_recursive


# @dataclass
PrimitiveStructure = Union[PolymerTopologyGraph, Atom, None]
class BadPrimitiveStructure(TypeError):
    '''Exception raised when a Primitive is initialized with an invalid structure'''
    ...

class Primitive:
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
            stereo_info : Optional[StereoInfo]=None,
            metadata : Optional[dict[Hashable, Any]]=None,
        ) -> None:
            # essential structural information
            self._structure = structure  # connection of internal parts (or lack thereof); used to find children in multiscale hierarchy
            self.ports = ports or []    # a collection of sites representing bonds to other Primitives
            self.shape = shape          # a rigid shape which approximates and abstracts the behavoir of the primitive in space

            # additional descriptors
            self.label = label                      # a handle for users to identify and distinguish Primitives by
            self.stereo_info = stereo_info or {}    # additional info about stereogenic atoms or bonds, if applicable
            self.metadata = metadata or {}          # literally any other information the user may want to bind to this Primitive
    
    # DEVNOTE: have platform-specific initializers/exporters be imported from interfaces (a la OpenFF Interchange)
    # @classmethod
    # def from_smiles(cls, smiles : str, positions : Optional[ndarray[Shape[N, 3], float]]=None) -> 'Primitive':
    #     '''Initialize a chemically-explicit Primitive from a SMILES string representation of the molecular graph'''
    #     ...
    # from_SMILES = from_smiles

    # @classmethod
    # def from_rdkit(cls, mol : Mol, conf_id : int=-1) -> 'Primitive':
    #     '''Initialize a chemically-explicit Primitive from an RDKit Mol'''
    #     ...
    
    # # export methods
    # def to_rdkit(self) -> Mol:
    #     '''Convert Primitive to an RDKit Mol'''
    #     ...      
       
    def copy(self) -> 'Primitive':
        '''Return a new Primitive with the same information as this one'''
        return Primitive(**self.__dict__) # TODO: deepcopy attributes dict?
    
    
    # properties derived from core components
    ## structural properties
    @property
    def structure(self) -> PrimitiveStructure:
        '''The internal chemical structure (or lack thereof) of this Primitive'''
        if not isinstance(self._structure, get_args(PrimitiveStructure)):
            raise TypeError(f'Primitive internal structure must be one of {get_args(PrimitiveStructure)}; got {type(self._structure)}')
        return self._structure
    
    @structure.setter
    def structure(self, new_structure : PrimitiveStructure) -> None:
        '''Set the internal chemical structure of this Primitive'''
        raise NotImplementedError
    
    @property
    def is_atomic(self) -> bool:
        '''Whether the Primitive at hand represents a single atom from the periodic table'''
        return isinstance(self.structure, Atom)
    
    @property
    def is_leaf(self) -> bool:
        '''Whether the Primitive at hand is at the bottom of a structural hierarchy'''
        return isinstance(self.structure, [Atom, None])
    
    @property
    def has_children(self) -> bool:
        '''Whether the Primitive at hand has children in a multiscale hierarchy'''
        return not self.is_leaf
    
    @property
    def is_all_atom(self) -> bool:
        '''Test if Primitive collectively represents a system defined to periodic table atom resolution'''
        if self.structure is None:
            return False
        elif isinstance(self.structure, Atom):
            return True
        elif isinstance(self.structure, PolymerTopologyGraph):
            return all(primitive.is_all_atom for primitive in self.structure)
        
    @property
    def num_atoms(self) -> int:
        '''Number of atoms the Primitive and its internal structure collectively represent'''
        # TODO: add ability to custom for advanced usage (e.g. indeterminate base CG chemistry?)
        if self.structure is None:
            return 0
        elif isinstance(self.structure, Atom):
            return 1
        elif isinstance(self.structure, PolymerTopologyGraph):
            _num_atoms : int = 0
            for subprimitive in self.structure:
                if not isinstance(subprimitive, Primitive):
                    raise TypeError(f'Primitive Topology improperly embedded; cannot determine number of atoms from non-Primitive {subprimitive}')
                _num_atoms += subprimitive.num_atoms
            return _num_atoms
    
    ## Shape properties
    ...
    
    ## Port properties
    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        return len(self.ports)
    
    @property
    def bondtype_inventory(self) -> Counter[BondType]:
        '''A Counter tracking the number of Ports of each BondType associated to this Primitive'''
        return Counter(port.bondtype for port in self.ports)
    
    @property
    def bondtype_index(self) -> tuple[tuple[int, int], ...]:
        '''
        Canonical identifier of all unique BondTypes by count among the Ports associated to this Primitive
        Consists of all (integer bondtype, count) pairs, sorted lexicographically
        '''
        return tuple(sorted(
            (int(bondtype), count)
                for (bondtype, count) in self.bondtype_inventory.items()
        ))
    
    
    # comparison methods
    # DEVNOTE: hashing needs to be stricter than equality, i.e. two Primitives may be indistinguishable by hash, but nevertheless equivalent
    ## canonical forms to be used for equivalence relations
    def _canonical_form_ports(self, separator : str=':', joiner : str='-') -> str:
        '''A canonical string representing this Primitive's ports'''
        return joiner.join(f'{BondType.values[bondtype_idx]}{separator}{count}' for (bondtype_idx, count) in self.bondtype_index)

    def _canonical_form_shape(self) -> str:
        '''A canonical string representing this Primitive's shape'''
        return type(self.shape).__name__ # still holds for NoneType

    def _canonical_form_structure(self) -> str:
        '''A canonical string representing this Primitive's structure'''
        raise NotImplementedError

    def canonical_form(self) -> str:
        '''
        Return a canonical string representation of the Primitive
        Two Primitives having the same canonical form are interchangable within a polymer system
        '''
        return f'{self._canonical_form_ports()}{self._canonical_form_shape()}{self._canonical_form_structure()}'
    
    def canonical_form_peppered(self) -> str:
        '''
        Return a canonical string representation of the Primitive with peppered metadata
        Used to distinguish two otherwise-equivalent Primitives, e.g. as needed for graph embedding
        
        Named for the cryptography technique of augmenting a hash by some external, stored data
        (as described in https://en.wikipedia.org/wiki/Pepper_(cryptography))
        '''
        return f'{self.canonical_form()}{self.label}' #{self.metadata}'

    ## dunders based on canonical and normal forms    
    def __str__(self) -> str:
        return f'{self.label}{self.functionality}{type(self.shape).__name__}{type(self.structure).__name__}'
    
    def __hash__(self):
        '''Hash used to compare Primitives for identity (NOT equivalence)'''
        # return hash(self.canonical_form())
        return hash(str(self))
    
    def __eq__(self, other : object) -> bool:
        # DEVNOTE: in order to use equivalent-but-not-identical Primitives as nodes in nx.Graph, __eq__ CANNOT evaluate similarity by hashes
        '''Check whether two primitives are equivalent (but not necessarily identical)'''
        if not isinstance(other, Primitive):
            raise TypeError(f'Cannot compare Primitive to {type(other)}')

        return self.canonical_form_peppered() == other.canonical_form_peppered()
    
    
    # resolution shift methods
    def atomize(self, uniquify: bool=False) -> Generator['Primitive', None, None]:
        '''Decompose primitive into its unique, constituent single-atom primitives'''
        ...
        
    def coagulate(self) -> 'Primitive':
        '''Combine all constituent primitives into a single, larger primitive'''
        ...
        
        
    # geometric methods
    def apply_rigid_transformation(self, transform : RigidTransform) -> 'Primitive': # TODO: make this specifically act on shape, ports, and structure?
        '''Apply an isometric (i.e. rigid) transformation to all parts of a Primitive which support it'''
        return Primitive(**apply_rigid_transformation_recursive(self.__dict__, transform))


@dataclass
class PrimitiveLexicon:
    '''Collection of primitives which form the basis of a'''
    primitives : dict[str, Primitive] = field(default_factory=dict) # primitives keyed by aliases
    
    # this will likely need **some** methods, but most of the sanitization ought to be the responsibility of the MID graph
    # this is more-or-less a glorified container for now
       