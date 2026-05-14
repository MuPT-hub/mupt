'''Typehints and shape enforcement for numpy arrays'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Literal,
    Optional,
    TypeVar,
    Union,
)
S = TypeVar('S') # pure generics
T = TypeVar('T') # pure generics

import numpy as np
import numpy.typing as npt
from numbers import Number, Real


# Numeric typehints
NumberLike = Union[np.number, Number, float] # DEV: stupidly, but "float" does not typehint as Number in static type checkers, so have to add it manually
Numeric = TypeVar('Numeric', bound=Number)
NumericNP = TypeVar('NumericNP', bound=np.dtype[np.number])
BoolNP = TypeVar('BoolNP', bound=np.dtype[np.bool_])

# RealValued = TypeVar('RealValued', bound=Real)
# RealValuedNP = TypeVar('RealValuedNP', bound=np.dtype[np.floating])

# Numpy array type annotations
Shape = tuple
DType = TypeVar('DType', bound=np.dtype)

## Typehints for indeterminate size of a given array dimension
M = TypeVar('M', bound=int) 
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)
Dims = TypeVar('Dims', bound=int) # intended to typehint the number of dimensions
DimsPlus = TypeVar('DimsPlus', bound=int) # intended to typehint the number of dimensions +1 (no easy way to do arithmetic to generic types yet)

# Fixed-size vector and array type annotations - consider deprecating, since they're not currently being used anywhere
## DEV: this type of hard-coding sucks, but is the best we can do with the current Python type system
Vector2  = np.ndarray[Shape[Literal[2]], NumericNP]
Vector3  = np.ndarray[Shape[Literal[3]], NumericNP]
Vector4  = np.ndarray[Shape[Literal[4]], NumericNP]
VectorN  = np.ndarray[Shape[N], NumericNP]

Array2x2 = np.ndarray[Shape[Literal[2], Literal[2]], NumericNP]
Array3x3 = np.ndarray[Shape[Literal[3], Literal[3]], NumericNP]
Array4x4 = np.ndarray[Shape[Literal[4], Literal[4]], NumericNP]

ArrayNx1 = np.ndarray[Shape[N, Literal[1]], NumericNP]
ArrayNx2 = np.ndarray[Shape[N, Literal[2]], NumericNP]
ArrayNx3 = np.ndarray[Shape[N, Literal[3]], NumericNP]
ArrayNx4 = np.ndarray[Shape[N, Literal[4]], NumericNP]

Array1xN = np.ndarray[Shape[Literal[1], N], NumericNP]
Array2xN = np.ndarray[Shape[Literal[2], N], NumericNP]
Array3xN = np.ndarray[Shape[Literal[3], N], NumericNP]
Array4xN = np.ndarray[Shape[Literal[4], N], NumericNP]

ArrayNxN = np.ndarray[Shape[N, N], NumericNP]
ArrayNxM = np.ndarray[Shape[N, M], NumericNP]
ArrayMxN = np.ndarray[Shape[M, N], NumericNP]

TriangulationIndices = np.ndarray[Shape[N, Literal[3]], np.dtype[np.integer]]
BitVectorN = np.ndarray[Shape[N], BoolNP]

# vector comparison
def as_n_vector(vectorlike : np.ndarray[Shape[N], DType], n : N=3) -> np.ndarray[Shape[N], DType]:
    '''Interpret array as a 1D n-element vector''' 
    if not isinstance(vectorlike, np.ndarray): # TODO: include support for list/tuple-like WITHOUT including sets, str, etc
        raise TypeError(f'Vectorlike must be a numpy array, not {type(vectorlike)}')
    if len(vectorlike) != n:
        raise ValueError(f'Expected {n}-element vectorlike, received {len(vectorlike)}-element array instead')
    
    return vectorlike.reshape(n)

def compare_optional_positions(
    position_1 : Optional[VectorN],
    position_2 : Optional[VectorN],
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
    
