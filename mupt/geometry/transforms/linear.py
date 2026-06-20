'''Application and construction of common linear transformations from planar normal vectors'''

import numpy as np
from ..arraytypes import Shape, Dims, NumericNP, Vector3, Array3x3


def projector(normal_vector : np.ndarray[Shape[Dims], NumericNP]) -> np.ndarray[Shape[Dims, Dims], NumericNP]:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector parallel to the normal vector provided

    Returns matrix which represents the parallel projector transformation
    '''
    (dim,) = normal_vector.shape # implicitly enforce 1D shape for vector
    return np.outer(normal_vector, normal_vector) / np.inner(normal_vector, normal_vector)
parallel = projector

def rejector(normal_vector : np.ndarray[Shape[Dims], NumericNP]) -> np.ndarray[Shape[Dims, Dims], NumericNP]:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector orthogonal to the normal vector provided

    Returns matrix which represents the orthogonal projector transformation
    '''
    (dim,) = normal_vector.shape # implicitly enforce 1D shape for vector
    return np.eye(dim, dtype=normal_vector.dtype) - projector(normal_vector) # equivalent to substracting parallel part off of vector transform is applied to
orthogonal = rejector

def reflector(normal_vector : np.ndarray[Shape[Dims], NumericNP]) -> np.ndarray[Shape[Dims, Dims], NumericNP]:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    reflects it across the plane defined by the normal vector provided
    
    Returns an orthogonal Householder matrix which represents the transformation
    '''
    (dim,) = normal_vector.shape # implicitly enforce 1D shape for vector
    return np.eye(dim, dtype=normal_vector.dtype) - 2*projector(normal_vector) # compute Householder reflection
householder = reflector

def orthogonalizer(normal_vector : Vector3) -> Array3x3:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    yields a vector orthogonal to both that vector and the normal vector provided here
    Equivalent to taking the cross product of v with the normal vector

    Returns matrix which represents the orthogonalization transformation
    '''
    (dims,) = normal_vector.shape # implicitly enforce 1D shape for vector
    return np.cross(
        np.eye(dims, dtype=normal_vector.dtype),
        normal_vector / np.linalg.norm(normal_vector),
    ) 
cross = orthogonalizer
