'''Indicators for coordinate axis-specific operations and indexing'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from enum import Enum
import numpy as np


def origin(dimension : int=3, dtype : type=float) -> np.ndarray:
    '''
    Return the origin in the specified number of dimensions
    '''
    _origin = np.zeros(dimension, dtype=dtype)
    _origin.setflags(write=False) # make immutable
    
    return _origin

ORIGIN2 = origin(2)
ORIGIN3 = origin(3)
ORIGIN4 = origin(4)

class CoordAxis(Enum):
    '''
    For making clear when a particular coordinate direction 
    is chosen for a task, particularly when indexing
    '''
    X = 0
    Y = 1
    Z = 2
    W = 3
    
    # lowercase aliases for the lazy
    x = 0
    y = 1
    z = 2
    w = 3
