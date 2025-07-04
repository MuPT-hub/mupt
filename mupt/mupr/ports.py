'''Abstractions of connections between two primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Generator, Optional
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, BondType

from ..geometry.arraytypes import Shape, Dims, DimsPlus
from ..geometry.transforms.linear import reflector, rotator
from ..geometry.transforms.affine import (
    translation,
    affine_matrix_from_linear_and_center,
    apply_affine_transformation_to_points,
)


# EXCEPTIONS
class MolPortError(Exception):
    '''Raised when port-related errors as encountered'''
    pass

class IncompatiblePortError(MolPortError):
    '''Raised when attempting to connect two Ports which are, for whatever reason, incompatible'''
    pass

# LINKER/BRIDGEHEAD QUERIES
LINKER_QUERY : str = '[#0X1]~*' # atomic number 0 (wild) attached to exactly 1 of anything (including possibly another wild-type atom)
# LINKER_QUERY = '[#0D1]~[!#0]' # neutronium-excluding linker query; requires that the linker be attached to a non-linker atom
LINKER_QUERY_MOL : Mol = Chem.MolFromSmarts(LINKER_QUERY)
# DEVNOTE: unclear whether X (total connections) or D (explicit connections) is the right choice for this query...
# ...or if there's ever a case where the two would not produce identical results; both seem to handle higher-order bonds correctly (i.e. treat double bond as "one" connection)

# PORT REPRESENTATION
@dataclass(frozen=False) # DEVNOTE need to preserve mutability for now, since coordinates of parts may change
class Port:
    '''Class for encapsulating the components of a "port" bonding site (linker-bond-bridgehead)'''
    linker     : Any
    bridgehead : Any
    bondtype : BondType = BondType.UNSPECIFIED
    linker_flavor : int = 0
    
    linker_position     : Optional[np.ndarray[Shape[Dims], float]] = None
    bridgehead_position : Optional[np.ndarray[Shape[Dims], float]] = None
    orientator_position : Optional[np.ndarray[Shape[Dims], float]] = None # TODO: validate this is orthogonal to the bond vector (if present)

    # initialization
    def copy(self) -> 'Port':
        '''Return a new Port with the same information as this one'''
        return Port(**self.__dict__)
    
    @classmethod
    def ports_from_rdkit(cls, mol : Mol, conf_id : int=-1) -> Generator['Port', None, None]:
        '''Determine all Ports contained in an RDKit Mol, as specified by wild-type linker atoms'''
        conformer = mol.GetConformer(conf_id) if (mol.GetNumConformers() > 0) else None
        for (linker_idx, bh_idx) in mol.GetSubstructMatches(LINKER_QUERY_MOL, uniquify=False): # DON'T de-duplify indices (fails to catch both ports on a neutronium)
            port = cls(
                linker=linker_idx, # for now, assign the index to allow easy reverse-lookup of the atom
                bridgehead=bh_idx,
                bondtype=mol.GetBondBetweenAtoms(bh_idx, linker_idx).GetBondType(),
                linker_flavor=mol.GetAtomWithIdx(linker_idx).GetIsotope(),
            )
            
            if conformer: # solicit coordinates, if available
                port.bridgehead_position = np.array(conformer.GetAtomPosition(bh_idx))
                port.linker_position     = np.array(conformer.GetAtomPosition(linker_idx))
                
                # TODO: offer option to make this more selective (i.e. choose which neghbor atom to use as stabilizer)
                for neighbor in mol.GetAtomWithIdx(bh_idx).GetNeighbors():
                    if neighbor.GetAtomicNum() > 0: # take first real neighbor atom as stabilizer
                        port.set_normal_from_stabilizer(stabilizer=conformer.GetAtomPosition(neighbor.GetIdx()))
                        break
                        
            yield port

    # comparison methods
    def __hash__(self) -> int:
        raise NotImplementedError
    
    def __eq__(self, other : 'Port') -> bool:
        raise NotImplementedError # criteria for bonding will be defined here
    
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
    def bond_vector(self) -> np.ndarray[Shape[Dims], float]:
        if not self.has_positions:
            raise ValueError
        return self.linker_position - self.bridgehead_position
    
    @property
    def bond_length(self) -> float:
        return np.linalg.norm(self.bond_vector)
    
    @property
    def unit_bond_vector(self) -> np.ndarray[Shape[Dims], float]:
        '''Unit vector in the same direction as the bond (oriented from bridgehead to linker)'''
        return self.bond_vector / self.bond_length
    
    def set_bond_length(self, new_bond_length : float) -> None:
        '''Move the linker site along the bond axis to a set distance away from the bridgehead'''
        self.linker_position = new_bond_length*self.unit_bond_vector + self.bridgehead_position
        
    ## normal vector (defines dihedral plane)
    @property
    def is_orientable(self) -> bool:
        '''Determine whether this port has a normal vector (i.e. dihedral plane orientation) defined'''
        return (self.orientator_position is not None)
    
    @property
    def normal_vector(self) -> np.ndarray[Shape[Dims], float]:
        # DEVNOTE: opted not to provide separate unit_normal_vector property,
        # since only the direction(not magnitude) matters for calculations
        '''Vector normal to the dihedral plane containing the bridgehead and linker'''
        if not self.is_orientable:
            raise ValueError('Port does not have a dihedral orientation set')
        
        normal = self.orientator_position - self.bridgehead_position
        normal /= np.linalg.norm(normal) 
        if not np.isclose(np.dot(self.bond_vector, normal), 0.0):
            raise ValueError('Badly set orientator position: resultant dihedral plane does not contain this Port\'s bond vector')
        
        return normal
    
    @normal_vector.setter
    def normal_vector(self, vector : np.ndarray[Shape[Dims], float]) -> None:
        # DEVNOTE: worth check that the vector pased is a valid normal? For now, outsourcing this to getter at calltime to minimize overhead
        '''Set dihedral plane normal vector'''
        self.orientator_position = vector + self.bridgehead_position

    def set_normal_from_stabilizer(self, stabilizer : np.ndarray[Shape[Dims], float]) -> None:
        '''Determine (and set) a unit vector normal to the plane containing the
        bridgehead, linker, and a third "stabilizer" point (provided as arg)'''
        self.normal_vector = np.cross(self.bond_vector, stabilizer - self.bridgehead_position)
        
    ## applying transformations
    # DEVNOTE: would like to use @optional_in_place here, but the current extend_to_methods mechanism works a little too well ("self" will NOT be passed as first arg to decorator)
    def affine_transformation(self, affine_matrix : np.ndarray[Shape[DimsPlus, DimsPlus], float], in_place : bool=False) -> Optional['Port']:
        '''Return a Port whose linker, bridgehead, and orientation positions
        (as provided) have been transformed by a given affine transformation matrix'''
        if not in_place:
            new_port = self.copy()
            new_port.affine_transformation(affine_matrix, in_place=True) # call in-place on the copy
            
            return new_port
            
        for attr in ('linker_position', 'bridgehead_position', 'orientator_position'):
            if (position := getattr(self, attr)) is not None:
                setattr(self, attr, apply_affine_transformation_to_points(position, affine_matrix))
    
    def alignment_transform_to(self, other : 'Port', dihedral_angle_rad : float=0.0) -> np.ndarray[Shape[DimsPlus, DimsPlus], float]:
        '''Compute an affine transformation which aligns this port with another port'''
        if not (self.has_positions and other.has_positions):
            raise ValueError('Cannot compute alignment transform with undefined Port positions')
        
        if not (self.is_orientable and other.is_orientable):
            raise ValueError('Cannot compute faithful orientation for alignment transform with undefined Port orientations')

        ## B reflects this Port's bond vector to the NEGATIVE of the other Port's bond vector (flips handedness)
        ## N aligns this Port's normal vector to the other Port's normal vector, after bond vector alignment (restores handedness)
        ## After bond and normal alignment, the dihedral angle is 0; R then sets it to the desired angle (negative due to transformed bond vectors being antiparallel)
        B = reflector(self.unit_bond_vector - (-other.unit_bond_vector)) # NOTE: deliberately didn't simplify negative sign to make action clearer
        N = reflector((B @ self.normal_vector) - other.normal_vector)
        D = rotator((B @ self.unit_bond_vector), -dihedral_angle_rad) # NOTE: don't need to apply N here, since the aligned bond vector lies within N's reflection plane
        
        return translation(*other.linker_position) \
            @ affine_matrix_from_linear_and_center(matrix=(D @ N @ B), center=None) \
            @ translation(*(-self.bridgehead_position))

    def align_to(self, other : 'Port', dihedral_angle_rad : float=0.0, match_bond_length : bool=False) -> None:
        '''Align this Port to another Port, based on the calculated alignment transform'''
        self.affine_transformation(
            affine_matrix=self.alignment_transform_to(other, dihedral_angle_rad),
            in_place=True,
        )
        if match_bond_length: 
            self.set_bond_length(other.bond_length) # ensure bond length matches the other port
        
    def aligned_to(self, other : 'Port', dihedral_angle_rad : float=0.0, match_bond_length : bool=False) -> 'Port':
        '''Return a copy of this Port aligned to Port "other"'''
        new_port = self.copy()
        new_port.align_to(other, dihedral_angle_rad=dihedral_angle_rad, match_bond_length=match_bond_length)

        return new_port
        