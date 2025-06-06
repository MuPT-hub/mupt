'''Representations and calculation methods for crystallographic unit cells, lattice parameters, and lattice coordinates'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generic, Literal
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .arraytypes import ndarray, Shape, Numeric


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
    def from_lattice_vectors(cls, lattice_vectors : ndarray[Shape[3, 3], float]) -> 'LatticeParameters':
        raise NotImplemented
    
    def to_lattice_vectors(self) -> ndarray[Shape[3, 3], float]:
        raise NotImplemented
    
# Coordinate subclasses
class Lattice(Coordinates[Real]): # NOTE: mbuild already has something like this
    '''For representing a periodic unit cell'''
    pass

class IntegralLattice(Coordinates[int]):
    '''For representing a lattice with integer-valued points'''
    pass
