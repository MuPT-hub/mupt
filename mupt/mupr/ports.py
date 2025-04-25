'''Abstractions of connections between two primitives'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Generator, Optional
from dataclasses import dataclass

import numpy as np
Shape = tuple # NOTE: yes, these could be imported from arraytypes, but seems silly to couple over such a small thing

from rdkit import Chem
from rdkit.Chem.rdchem import Mol, BondType


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
@dataclass(frozen=True)
class Port:
    '''Class for encapsulating the components of a "port" bonding site (linker-bond-bridgehead)'''
    linker : Any # TODO: nail down exactly what objects these should be to work in general (wihtout depending on definition of Primitives or Atoms)
    linker_flavor : int = 0
    linker_position : Optional[np.ndarray[Shape[3], float]] = None
    
    bondtype : BondType
    
    bridgehead : Any # TODO: nail down exactly what objects these should be to work in general (wihtout depending on definition of Primitives or Atoms)
    bridgehead_position : Optional[np.ndarray[Shape[3], float]] = None
    normal : Optional[np.ndarray[Shape[3], float]] = None 

    # initialization
    @classmethod
    def from_rdkit(cls, mol : Mol, conf_id : int=-1) -> Generator['Port', None, None]:
        '''Determine all Ports contained in an RDKit Mol, as specified by wild-type linker atoms'''
        conformer = mol.GetConformer(conf_id) if (mol.GetNumConformers() > 0) else None
        for (linker_idx, bh_idx) in mol.GetSubstructMatches(LINKER_QUERY_MOL, uniquify=False): # DON'T de-duplify indices (fails to catch both ports on a neutronium)
            port = cls(
                linker=linker_idx, # for now, assign the index to allow easy reverse-lookup of the atom
                linker_flavor=mol.GetAtomWithIdx(linker_idx).GetIsotope(),
                bondtype=mol.GetBondBetweenAtoms(bh_idx, linker_idx).GetBondType(),
                bridgehead=bh_idx,
            )
            
            if conformer: # solicit coordinates, if available
                port.bridgehead_coord = np.array(conformer.GetAtomPosition(bh_idx))
                port.linker_coord     = np.array(conformer.GetAtomPosition(linker_idx))
                # for neighbor in bh_atom.GetNeighbors():
                #     if neighbor.GetAtomicNum() > 0: # take first real neighbor atom as stabilizer
                #         port.set_normal_from_stabilizer(stabilizer=conformer.GetAtomPosition(neighbor.GetIdx()))
                #         break
                        
            yield port

    # comparison methods
    def __hash__(self) -> int:
        raise NotImplemented
    
    def __eq__(self, other : 'Port') -> bool:
        raise NotImplemented # criteria for bonding will be defined here
    
    @classmethod
    def is_bondable_to(cls, port1 : 'Port', port2 : 'Port') -> bool:
        '''Determine whether two ports are bondable to each other'''
        raise NotImplemented
    
    # geometric properties
    @property
    def has_coords(self) -> bool:
        return not ((self.bridgehead_coord is None) or (self.linker_coord is None))
    
    @property
    def bond_vector(self) -> np.ndarray[Shape[3], float]:
        if not self.has_coords:
            raise ValueError
        return self.linker_coord - self.bridgehead_coord
    
    @property
    def bond_length(self) -> float:
        return np.linalg.norm(self.bond_vector)
    
    @property
    def unit_bond_vector(self) -> np.ndarray[Shape[3], float]:
        '''Unit vector in the same direction as the bond (oriented from bridgehead to linker)'''
        return self.bond_vector / self.bond_length
    
    def set_bond_length(self, new_bond_length : float) -> None:
        '''Move the linker site along the bond axis to a set distance away from the bridgehead'''
        self.linker_coord = new_bond_length*self.unit_bond_vector + self.bridgehead_coord
        
    def set_normal_from_stabilizer(self, stabilizer : np.ndarray[Shape[3], float]) -> None:
        '''Determine (and set) a unit vector normal to the plane containing the
        bridgehead, linker, and a third "stabilizer" point (provided as arg)'''
        normal = np.cross(self.bond_vector, stabilizer - self.bridgehead_coord)
        unit_normal = normal / np.linalg.norm(normal)
        
        self.normal = unit_normal