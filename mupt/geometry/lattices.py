'''Representations and calculation methods for crystallographic unit cells, lattice parameters, and lattice coordinates'''


from typing import Generic
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .arraytypes import Numeric, Array3x3


class Coordinates(Generic[Numeric]):
    '''Encapsulation class for storing sets of coordinates and performing transfomations on those coordinates'''
    pass

@dataclass
class LatticeParameters: # TODO : incorporate unit-awareness
    '''For parameterizing a single crystallographic unit cell'''
    a : float
    b : float
    c : float

    alpha : float = field(default=np.pi / 2) # make cell orthorhombic by default
    beta  : float = field(default=np.pi / 2) # make cell orthorhombic by default
    gamma : float = field(default=np.pi / 2) # make cell orthorhombic by default
    
    @classmethod
    def from_lattice_vectors(cls, lattice_vectors : Array3x3) -> 'LatticeParameters':
        raise NotImplemented
    
    def to_lattice_vectors(self) -> Array3x3:
        raise NotImplemented
    
# Coordinate subclasses
class Lattice(Coordinates[Real]): # NOTE: mbuild already has something like this
    '''For representing a periodic unit cell'''
    pass

class IntegralLattice(Coordinates[int]):
    '''For representing a lattice with integer-valued points'''
    pass
