'''Typehints specific to numpy and other array-related functionality'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Annotated, Generic, Literal, TypeVar

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
DimsPlus = TypeVar('Dims', bound=int) # intended to typehint the number of dimensions +1 (no easy way to do arithmetic to generic types yet)
M = TypeVar('M', bound=int) # typehint the size of a given dimension
N = TypeVar('N', bound=int) # typehint the size of a given dimension
P = TypeVar('P', bound=int) # typehint the size of a given dimension

# Fixed-size vector and array type annotations
## DEV: this type of hard-coding sucks, but is the best we can do with the current Python type system
Vector2  = Annotated[npt.NDArray[DType], Shape[2]]
Array2x2 = Annotated[npt.NDArray[DType], Shape[2, 2]]
ArrayNx2 = Annotated[npt.NDArray[DType], Shape[N, 2]]

Vector3  = Annotated[npt.NDArray[DType], Shape[3]]
Array3x3 = Annotated[npt.NDArray[DType], Shape[3, 3]]
ArrayNx3 = Annotated[npt.NDArray[DType], Shape[N, 3]]

Vector4  = Annotated[npt.NDArray[DType], Shape[4]]
Array4x4 = Annotated[npt.NDArray[DType], Shape[4, 4]]
ArrayNx4 = Annotated[npt.NDArray[DType], Shape[N, 4]]

VectorN  = Annotated[npt.NDArray[DType], Shape[N]]
ArrayNxN = Annotated[npt.NDArray[DType], Shape[N, N]]
ArrayNxM = Annotated[npt.NDArray[DType], Shape[N, M]]
ArrayMxN = Annotated[npt.NDArray[DType], Shape[M, N]]
