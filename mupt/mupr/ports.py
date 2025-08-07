'''Abstractions of connections between two primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, ClassVar, Literal, Optional
Shape = tuple # alias for typehinting array shapes
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform
from rdkit.Chem.rdchem import BondType

from ..geometry.measure import normalized
from ..geometry.transforms.linear import rejector
from ..geometry.transforms.rigid.rotations import alignment_rotation


class MolPortError(Exception):
    '''Raised when port-related errors as encountered'''
    pass

class IncompatiblePortError(MolPortError):
    '''Raised when attempting to connect two Ports which are, for whatever reason, incompatible'''
    pass


@dataclass(frozen=False) # DEVNOTE need to preserve mutability for now, since coordinates of parts may change
class Port:
    '''Abstraction of the notion of a chemical bond between a known body (anchor) and an indeterminate neghbor body (linker)'''
    # DEVNOTE: want to hone in on the allowable types for these (Hashable?)
    anchor : Any
    linker : Any
    bondtype : BondType = BondType.UNSPECIFIED
    
    query_smarts : str = ''
    
    linker_position : Optional[np.ndarray[Shape[Literal[3]], float]] = None
    anchor_position : Optional[np.ndarray[Shape[Literal[3]], float]] = None
    tangent_position : Optional[np.ndarray[Shape[Literal[3]], float]] = None
    
    # DEVNOTE: this will need updating if more position-type attributes are added; manually curating this is fine for now
    _POSITION_ATTRS : ClassVar[tuple[str]] = ('anchor_position', 'linker_position', 'tangent_position') 

    # initialization
    def copy(self) -> 'Port':
        '''Return a new Port with the same information as this one'''
        return Port(**self.__dict__)


    # comparison methods
    def canonical_form(self) -> BondType:
        '''Return a canonical form used to distinguish equivalent Ports'''
        return self.bondtype # TODO: make this more descriptive; good enough for now
    
    def __hash__(self) -> int:
        raise NotImplementedError # DEVNOTE: need to decide what info should (and shouldn't) go into the making of this sausage
    
    # def __eq__(self, other : 'Port') -> bool:
        # return hash(self) == hash(other)

    def bondable_with(self, other : 'Port') -> bool:
        '''Determine whether this Port is bondable with another Port'''
        if not isinstance(other, Port):
            return False # DEVNOTE: raise TypeError instead (or at least log a warning)?
        
        return (
            (self.linker == other.anchor) # DEVNOTE: might opt for more general binary relation in the future
            and (self.anchor == other.linker)
            and (self.bondtype == other.bondtype)
        )
    
    @staticmethod
    def compare_optional_positions(
        position_1 : Optional[np.ndarray[Shape[Any], float]],
        position_2 : Optional[np.ndarray[Shape[Any], float]],
        **kwargs,
    ) -> bool:
        '''Check that two positional attributes are either 1) both undefined, or 2) both defined and equal'''
        if type(position_1) != type(position_2):
            return False
        
        if position_1 is None: # both are None
            return True
        elif isinstance(position_1, np.ndarray):
            return np.allclose(position_1, position_2, **kwargs)
        else:
            raise TypeError(f'Expected position attributes to be either None or numpy.ndarray, got {type(position_1)} and {type(position_2)}')
    
    def coincides_with(self, other : 'Port') -> bool:
        '''Determine whether this Port coincides with another Port'''
        return all(
            self.compare_optional_positions(getattr(self, position_attr), getattr(other, position_attr))
                for position_attr in self._POSITION_ATTRS
        )

    
    # geometric properties
    @property
    def has_positions(self) -> bool:
        return (self.anchor_position is not None) and (self.linker_position is not None)
    
    @property
    def has_defined_dihedral_plane(self) -> bool:
        '''Determine whether this port has a tangent vector (i.e. dihedral plane orientation) defined'''
        return (self.tangent_position is not None)
    
    ## bond vector
    @property
    def bond_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        if not self.has_positions:
            raise ValueError
        return self.linker_position - self.anchor_position
    
    @property
    def bond_length(self) -> float:
        return np.linalg.norm(self.bond_vector)
    
    @property
    def unit_bond_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        '''Unit vector in the same direction as the bond (oriented from anchor to linker)'''
        return self.bond_vector / self.bond_length
    
    def set_bond_length(self, new_bond_length : float) -> None:
        '''Move the linker site along the bond axis to a set distance away from the anchor'''
        self.linker_position = new_bond_length*self.unit_bond_vector + self.anchor_position
        
    ## tangent vector (defines dihedral plane)
    @property
    def tangent_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        '''
        Vector tangent to the dihedral plane and orthogonal to the bond vector
        
        The tangent and bond vectors span the dihedral plane and 
        fix a local right-handed coordinate system for the Port
        '''
        if not self.has_defined_dihedral_plane:
            raise ValueError('Port does not have a dihedral orientation set')
        
        tangent = normalized(self.tangent_position - self.anchor_position) # DEVNOTE: worth providing option to not normalize?
        if not np.isclose(np.dot(self.bond_vector, tangent), 0.0):
            raise ValueError('Badly set tangent position: resultant dihedral plane does not contain this Port\'s bond vector')
        
        return tangent
    
    @tangent_vector.setter
    def tangent_vector(self, vector : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Update tangent positions given a new tangent vector'''
        # NOTE: implemented this way to get tangent to transform correctly under rigid transformations;
        # a DIFFERENCE between vectors is invariant to shifts of the origin; the same is done for the bond vector
        self.tangent_position = vector + self.anchor_position

    def set_tangent_from_coplanar_point(self, coplanar_point : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Set the dihedral tangent point from a third point in the dihedral plane'''
        self.tangent_vector = rejector(self.bond_vector) @ (coplanar_point - self.anchor_position)

    def set_tangent_from_normal_point(self, normal_point : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Set the dihedral tangent point from a point on the span of the normal to the dihedral plane'''
        self.tangent_vector = np.cross(self.bond_vector, normal_point - self.anchor_position)
        
    ## applying transformations
    # DEVNOTE: would like to use @optional_in_place here, but the current extend_to_methods mechanism works a little too well ("self" will NOT be passed as first arg to decorator)
    def apply_rigid_transformation(self, transform : RigidTransform, in_place : bool=False) -> Optional['Port']:
        '''Return a Port whose anchor, linker, and orientation positions
        (if provided) have been transformed by a given rigid transformation'''
        if not in_place:
            new_port = self.copy()
            new_port.apply_rigid_transformation(transform, in_place=True) # call in-place on the copy
            
            return new_port
            
        for attr in self._POSITION_ATTRS:
            if (position := getattr(self, attr)) is not None:
                setattr(self, attr, transform.apply(position))
    
    def alignment_transform_to(self, other : 'Port', dihedral_angle_rad : float=0.0) -> RigidTransform:
        '''
        Compute an isometric (i.e. rigid) transformation which aligns a pair of Ports by making
        the linker point of this Port coincident with the anchor of the other Port,
        the Ports' bond vectors antiparallel, and the Ports' tangent vectors subtend the
        desired dihedral angle in radians (by default, 0.0 rad)
        
        If the two Ports have the same bond length, the anchor of this Port will be coincident with the linker
        of the other; otherwise, the anchor will merely lay on the span of the other Ports bond vector
        '''
        if not (self.has_positions and other.has_positions):
            raise ValueError('Cannot compute alignment transform with undefined Port positions')
        
        if not (self.has_defined_dihedral_plane and other.has_defined_dihedral_plane):
            raise ValueError('Cannot compute faithful orientation for alignment transform with undefined Port orientations')

        ## NOTE: the orthogonality of the tangent and bond vector of each Port allows the tangent alignment to
        ## not disturb the preceding bond antialignment, and to fix a unique orthogonal change of basis
        bond_antialignment : Rotation = alignment_rotation(self.unit_bond_vector, -other.unit_bond_vector)
        tangent_alignment  : Rotation = alignment_rotation(bond_antialignment.apply(self.tangent_vector), other.tangent_vector)
        dihedral_rotation = Rotation.from_rotvec(dihedral_angle_rad * other.unit_bond_vector)

        return ( # order of application of operations reads bottom-to-top (rightmost operator acts first)
            RigidTransform.from_translation(other.linker_position)
            * RigidTransform.from_rotation(dihedral_rotation)
            * RigidTransform.from_rotation(tangent_alignment)
            * RigidTransform.from_rotation(bond_antialignment)
            * RigidTransform.from_translation(-self.anchor_position)
        )

    def align_to(self, other : 'Port', dihedral_angle_rad : float=0.0, match_bond_length : bool=False) -> None:
        '''Align this Port to another Port, based on the calculated alignment transform'''
        self.apply_rigid_transformation(
            transform=self.alignment_transform_to(other, dihedral_angle_rad),
            in_place=True,
        )
        if match_bond_length: 
            self.set_bond_length(other.bond_length) # ensure bond length matches the other port
        
    def aligned_to(self, other : 'Port', dihedral_angle_rad : float=0.0, match_bond_length : bool=False) -> 'Port':
        '''Return a copy of this Port aligned to Port "other"'''
        new_port = self.copy()
        new_port.align_to(other, dihedral_angle_rad=dihedral_angle_rad, match_bond_length=match_bond_length)

        return new_port
        