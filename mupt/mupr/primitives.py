'''Information classes for sets of polymer unit primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Generator, Hashable, Optional, Union, get_args
from dataclasses import dataclass, field
from enum import Enum # consider having a bitwise Enum to encode possible specification states of a primitive??

from scipy.spatial.transform import RigidTransform

from rdkit.Chem.rdchem import (
    Atom,
    Bond,
    Mol,
    # DEVNOTE: not yet sure what the best way to represent stereochemistry is
    StereoInfo,
    StereoType,
    StereoDescriptor,
    ChiralType,
)

from .ports import Port
from .topology import PolymerTopologyGraph

from ..geometry.arraytypes import ndarray, N, Shape
from ..geometry.shapes import BoundedShape, PointCloud
from ..geometry.transforms.rigid import apply_rigid_transformation_recursive


# @dataclass
type PrimitiveStructure = Union[PolymerTopologyGraph, Atom, None]
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

    def __str__(self) -> str:
        pass
    
    ## comparison methods    
    def __hash__(self):
        # TODO: include information about content (not just number) of Ports, find way to make distinguishable
        return hash(f'{self.label}{self.functionality}{type(self.shape).__name__}{type(self.shape).__name__}')
    
    # DEVNOTE: in order to use equivalent-but-not-identical Primitives as nodes in nx.Graph, __eq__ CANNOT evaluate similarity by hashes
    def __eq__(self, other : object) -> bool:
        '''Check whether two primitives are equivalent (but not necessarily identical)'''
        if not isinstance(other, Primitive):
            raise TypeError(f'Cannot compare Primitive to {type(other)}')
        return NotImplemented
       
    # properties
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
        '''Test if the primitive at hand represents a single atom from the periodic table'''
        return isinstance(self.structure, Atom)
    
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
        if self.num_atoms is not None:
            return 0
        elif self.is_atomic:
            return 1
        elif isinstance(self.structure, PolymerTopologyGraph):
            _num_atoms : int = 0
            for sum_primitive in self.structure:
                if not isinstance(sum_primitive, Primitive):
                    raise TypeError(f'Primitive Topology improperly embedded; cannot determine number of atoms from non-Primitive {sum_primitive}')
                _num_atoms += sum_primitive.num_atoms
            return _num_atoms

    @property
    def functionality(self) -> int:
        '''Number of neighboring primitives which can be attached to this primitive'''
        return len(self.ports)
    
    # initialization methods         
    def copy(self) -> 'Primitive':
        '''Return a new Primitive with the same information as this one'''
        return Primitive(**self.__dict__)

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
       