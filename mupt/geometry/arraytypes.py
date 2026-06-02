'''Typehints and shape enforcement for numpy arrays'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Annotated,
    Generic,
    Literal,
    Optional,
    TypeVar,
)
S = TypeVar('S') # pure generics
T = TypeVar('T') # pure generics

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from numbers import Number, Real


# Numeric typehints
Numeric = TypeVar('Numeric', bound=Number) # typehint a number-like generic type
RealValued = TypeVar('RealValued', bound=Real)

# Numpy array type annotations
Shape = tuple # the shape field of a numpy array
DType = TypeVar('DType', bound=np.generic) # the data type of a numpy array

Dims = TypeVar('Dims', bound=int) # intended to typehint the number of dimensions
DimsPlus = TypeVar('DimsPlus', bound=int) # intended to typehint the number of dimensions +1 (no easy way to do arithmetic to generic types yet)
M = TypeVar('M', bound=int) # typehint the size of a given dimension
N = TypeVar('N', bound=int) # typehint the size of a given dimension
P = TypeVar('P', bound=int) # typehint the size of a given dimension

# Fixed-size vector and array type annotations - consider deprecating, since they're not currently being used anywhere
## DEV: this type of hard-coding sucks, but is the best we can do with the current Python type system
Vector2  = np.ndarray[Shape[Literal[2]], Numeric]
Vector3  = np.ndarray[Shape[Literal[3]], Numeric]
Vector4  = np.ndarray[Shape[Literal[4]], Numeric]
VectorN  = np.ndarray[Shape[N], Numeric]

Array2x2 = np.ndarray[Shape[Literal[2], Literal[2]], Numeric]
Array3x3 = np.ndarray[Shape[Literal[3], Literal[3]], Numeric]
Array4x4 = np.ndarray[Shape[Literal[4], Literal[4]], Numeric]

ArrayNx2 = np.ndarray[Shape[N, Literal[2]], Numeric]
ArrayNx3 = np.ndarray[Shape[N, Literal[3]], Numeric]
ArrayNx4 = np.ndarray[Shape[N, Literal[4]], Numeric]

Array2xN = np.ndarray[Shape[Literal[2], N], Numeric]
Array3xN = np.ndarray[Shape[Literal[3], N], Numeric]
Array4xN = np.ndarray[Shape[Literal[4], N], Numeric]

ArrayNxN = np.ndarray[Shape[N, N], Numeric]
ArrayNxM = np.ndarray[Shape[N, M], Numeric]
ArrayMxN = np.ndarray[Shape[M, N], Numeric]


# vector comparison
def as_n_vector(vectorlike : np.ndarray[Shape[N], DType], n : N=3) -> np.ndarray[Shape[N], DType]:
    '''Interpret array as a 1D n-element vector''' 
    if not isinstance(vectorlike, np.ndarray): # TODO: include support for list/tuple-like WITHOUT including sets, str, etc
        raise TypeError(f'Vectorlike must be a numpy array, not {type(vectorlike)}')
    if len(vectorlike) != n:
        raise ValueError(f'Expected {n}-element vectorlike, received {len(vectorlike)}-element array instead')
    
    return vectorlike.reshape(n)

def compare_optional_positions(
    position_1 : Optional[np.ndarray[Shape[N], float]],
    position_2 : Optional[np.ndarray[Shape[N], float]],
    **kwargs,
) -> bool:
    '''Check that two positional values are either 1) both undefined, or 2) both defined and equal'''
    # DEV: replace with monadic interface down the line ("Maybe" pattern?)
    if type(position_1) != type(position_2):
        return False
    
    if position_1 is None: # both are None
        return True
    elif isinstance(position_1, np.ndarray):
        return np.allclose(position_1, position_2, **kwargs)
    else:
        raise TypeError(f'Expected positions to be either None or numpy.ndarray, got {type(position_1)} and {type(position_2)}')
    
