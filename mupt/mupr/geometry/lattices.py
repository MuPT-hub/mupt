'''Representations and calculation methods for crystallographic unit cells, lattice parameters, and lattice coordinates'''

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .coordinates import Coordinates
from ...mutils.arraytypes import Shape, Array3x3


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
    def from_lattice_vectors(self, lattice_vectors : Array3x3) -> 'LatticeParameters':
        raise NotImplemented
    
    def to_lattice_vectors(self) -> Array3x3:
        raise NotImplemented
    
# Coordinate subclasses
class IntegralLattice(Coordinates[int]):
    '''For representing a lattice with integer-valued points'''
    pass

class Lattice(Coordinates[Real]): # NOTE: mbuild already has something like this
    '''For representing a periodic unit cell'''
    pass