'''Typehints and shape enforcement for numpy arrays'''


from typing import (
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
# pure generics
S = TypeVar('S')
T = TypeVar('T')

import numpy as np
import numpy.typing as npt
from numbers import Number#, Real


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

## types accepted by 'order' arg of np.linalg.norm()
OrderType = Optional[Union[int, Literal['fro'], Literal['nuc']]] 

## Typehints for indeterminate size of a given array dimension
M = TypeVar('M', bound=int) 
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)
Dims = TypeVar('Dims', bound=int) # intended to typehint the number of dimensions
DimsPlus = TypeVar('DimsPlus', bound=int) # intended to typehint the number of dimensions +1 (no easy way to do arithmetic to generic types yet)

# Fixed-size vector and array type annotations - consider deprecating, since they're not currently being used anywhere
## TB DEV: this type of hard-coding sucks, but is the best we can do with the current Python type system
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
def as_n_vector(
    vectorlike : VectorN | Array1xN | ArrayNx1 | Sequence[NumberLike],
    dimension : Optional[int]=None,
    dtype : Optional[npt.DTypeLike]=None,
) -> VectorN:
    '''
    Convert row vector, column vector, Nx1 array, or 1xN array into
    1D column vector with appropriate dimension and data type
    
    Enables permissive ingestion of vector-shaped objects
    '''
    # N.B.: strings and byte-like are TECHNICALLY also Sequences, but not the kind we want here
    if isinstance(vectorlike, (str, bytes)) \
        or (not isinstance(vectorlike, (np.ndarray, Sequence))):
        raise TypeError(f'Vectorlike must be a numpy array of Sequence of Numerics, not {type(vectorlike).__name__}')
    
    vector_column = np.atleast_2d(vectorlike).reshape(-1) # permits transposed and nested vector inputs
    if (dimension is not None) and (vector_column.shape != (dimension,)):
        raise ValueError(
            f'Expected vector with shape {(dimension,)}, got {vector_column.shape}'
        )
    
    if dtype is not None:
        vector_column = vector_column.astype(dtype)
        
    return vector_column
