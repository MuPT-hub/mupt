'''Abstractions of connections between two primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Generator, Literal, Optional
Shape = tuple # alias for typehinting array shapes
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, Mol, BondType

from ..geometry.measure import normalized
from ..geometry.transforms.linear import rejector
from ..geometry.transforms.rigid.rotations import alignment_rotation
from ..chemistry.linkers import LINKER_QUERY_MOL


class MolPortError(Exception):
    '''Raised when port-related errors as encountered'''
    pass

class IncompatiblePortError(MolPortError):
    '''Raised when attempting to connect two Ports which are, for whatever reason, incompatible'''
    pass


@dataclass(frozen=False) # DEVNOTE need to preserve mutability for now, since coordinates of parts may change
class Port:
    '''Class for encapsulating the components of a "port" bonding site (linker-bond-bridgehead)'''
    linker     : Any
    bridgehead : Any
    
    bondtype : BondType = BondType.UNSPECIFIED
    linker_flavor : int = 0
    query_smarts : str = ''
    
    linker_position     : Optional[np.ndarray[Shape[Literal[3]], float]] = None
    bridgehead_position : Optional[np.ndarray[Shape[Literal[3]], float]] = None
    tangent_position : Optional[np.ndarray[Shape[Literal[3]], float]] = None # TODO: validate this is orthogonal to the bond vector (if present)

    # initialization
    def copy(self) -> 'Port':
        '''Return a new Port with the same information as this one'''
        return Port(**self.__dict__)
    
    @classmethod
    def ports_from_rdkit(cls, mol : Mol, conf_id : int=-1) -> Generator['Port', None, None]:
        '''Determine all Ports contained in an RDKit Mol, as specified by wild-type linker atoms'''
        conformer = mol.GetConformer(conf_id) if (mol.GetNumConformers() > 0) else None
        for (linker_idx, bh_idx) in mol.GetSubstructMatches(LINKER_QUERY_MOL, uniquify=False): # DON'T de-duplify indices (fails to catch both ports on a neutronium)
            linker_atom : Atom = mol.GetAtomWithIdx(linker_idx)
            port_bond   : Bond = mol.GetBondBetweenAtoms(bh_idx, linker_idx)
            
            port = cls(
                linker=linker_idx, # for now, assign the index to allow easy reverse-lookup of the atom
                bridgehead=bh_idx,
                bondtype=port_bond.GetBondType(),
                linker_flavor=linker_atom.GetIsotope(),
                query_smarts=Chem.MolFragmentToSmarts(
                    mol,
                    atomsToUse=[linker_idx, bh_idx],
                    bondsToUse=[port_bond.GetIdx()],
                )
            )
            
            if conformer: # solicit coordinates, if available
                port.linker_position     = np.array(conformer.GetAtomPosition(linker_idx))
                port.bridgehead_position = np.array(conformer.GetAtomPosition(bh_idx))
                
                # TODO: offer option to make this more selective (i.e. choose which neighbor atom lies in the dihedral plane)
                for neighbor in mol.GetAtomWithIdx(bh_idx).GetNeighbors():
                    if neighbor.GetAtomicNum() > 0: # take first real neighbor atom for now
                        port.set_tangent_from_coplanar_point(conformer.GetAtomPosition(neighbor.GetIdx()))
                        break
                        
            yield port

    # comparison methods
    def __hash__(self) -> int:
        raise NotImplementedError # DEVNOTE: need to decide what info should (and shouldn't) go into the making of this sausage
    
    # def __eq__(self, other : 'Port') -> bool:
        # return hash(self) == hash(other)

    @classmethod
    def is_bondable_to(cls, port1 : 'Port', port2 : 'Port') -> bool:
        '''Determine whether two ports are bondable to each other'''
        raise NotImplementedError
    
    # geometric properties
    @property
    def has_positions(self) -> bool:
        return (self.bridgehead_position is not None) and (self.linker_position is not None)
    
    ## bond vector
    @property
    def bond_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        if not self.has_positions:
            raise ValueError
        return self.linker_position - self.bridgehead_position
    
    @property
    def bond_length(self) -> float:
        return np.linalg.norm(self.bond_vector)
    
    @property
    def unit_bond_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        '''Unit vector in the same direction as the bond (oriented from bridgehead to linker)'''
        return self.bond_vector / self.bond_length
    
    def set_bond_length(self, new_bond_length : float) -> None:
        '''Move the linker site along the bond axis to a set distance away from the bridgehead'''
        self.linker_position = new_bond_length*self.unit_bond_vector + self.bridgehead_position
        
    ## tangent vector (defines dihedral plane)
    @property
    def has_defined_dihedral_plane(self) -> bool:
        '''Determine whether this port has a tangent vector (i.e. dihedral plane orientation) defined'''
        return (self.tangent_position is not None)
    
    @property
    def tangent_vector(self) -> np.ndarray[Shape[Literal[3]], float]:
        '''
        Vector tangent to the dihedral plane and orthogonal to the bond vector
        
        The tangent and bond vectors span the dihedral plane and 
        fix a local right-handed coordinate system for the Port
        '''
        if not self.has_defined_dihedral_plane:
            raise ValueError('Port does not have a dihedral orientation set')
        
        tangent = normalized(self.tangent_position - self.bridgehead_position) # DEVNOTE: worth providing option to not normalize?
        if not np.isclose(np.dot(self.bond_vector, tangent), 0.0):
            raise ValueError('Badly set tangent position: resultant dihedral plane does not contain this Port\'s bond vector')
        
        return tangent
    
    @tangent_vector.setter
    def tangent_vector(self, vector : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Update tangent positions given a new tangent vector'''
        # NOTE: implemented this way to get tangent to transform correctly under rigid transformations;
        # a DIFFERENCE between vectors is invariant to shifts of the origin; the same is done for the bond vector
        self.tangent_position = vector + self.bridgehead_position

    def set_tangent_from_coplanar_point(self, coplanar_point : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Set the dihedral tangent point from a third point in the dihedral plane'''
        self.tangent_vector = rejector(self.bond_vector) @ (coplanar_point - self.bridgehead_position)

    def set_tangent_from_normal_point(self, normal_point : np.ndarray[Shape[Literal[3]], float]) -> None:
        '''Set the dihedral tangent point from a point on the span of the normal to the dihedral plane'''
        self.tangent_vector = np.cross(self.bond_vector, normal_point - self.bridgehead_position)
        
    ## applying transformations
    # DEVNOTE: would like to use @optional_in_place here, but the current extend_to_methods mechanism works a little too well ("self" will NOT be passed as first arg to decorator)
    def apply_rigid_transformation(self, transform : RigidTransform, in_place : bool=False) -> Optional['Port']:
        '''Return a Port whose linker, bridgehead, and orientation positions
        (if provided) have been transformed by a given rigid transformation'''
        if not in_place:
            new_port = self.copy()
            new_port.apply_rigid_transformation(transform, in_place=True) # call in-place on the copy
            
            return new_port
            
        for attr in ('linker_position', 'bridgehead_position', 'tangent_position'):
            if (position := getattr(self, attr)) is not None:
                setattr(self, attr, transform.apply(position))
    
    def alignment_transform_to(self, other : 'Port', dihedral_angle_rad : float=0.0) -> RigidTransform:
        '''
        Compute an isometric (i.e. rigid) transformation which aligns a pair of Ports by making
        the linker point of this Port coincident with the bridgehead of the other Port,
        the Ports' bond vectors antiparallel, and the Ports' tangent vectors subtend the
        desired dihedral angle in radians (by default, 0.0 rad)
        
        If the two Ports have the same bond length, the bridgehead of this Port will be coincident with the linker
        of the other; otherwise, the bridgehead will merely lay on the span of the other Ports bond vector
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
            * RigidTransform.from_translation(-self.bridgehead_position)
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
        