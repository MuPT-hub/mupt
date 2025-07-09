'''Application and construction of common linear transformations'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from scipy.spatial.transform import Rotation

from ..arraytypes import Shape, N, Dims, Numeric


def projector(direction_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector parallel to the direction vector provided
    
    Returns matrix which represents the parallel projector transformation
    '''
    (dim,) = direction_vector.shape # implicitly enforce 1D shape for vector
    return np.outer(direction_vector, direction_vector) / np.inner(direction_vector, direction_vector)
parallel = projector

def rejector(direction_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector orthogonal to the direction vector provided
    
    Returns matrix which represents the orthogonal projector transformation
    '''
    (dim,) = direction_vector.shape # implicitly enforce 1D shape for vector
    return np.eye(dim, dtype=direction_vector.dtype) - projector(direction_vector) # equivalent to substracting parallel part off of vector transform is applied to
orthogonal = rejector

def reflector(normal_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    reflects it across the plane defined by the normal vector provided
    
    Returns an orthogonal Householder matrix which represents the transformation
    '''
    (dim,) = normal_vector.shape # implicitly enforce 1D shape for vector
    return np.eye(dim, dtype=normal_vector.dtype) - 2*projector(normal_vector) # compute Householder reflection
householder = reflector

def orthogonalizer(direction_vector : np.ndarray[Shape[3], Numeric]) -> np.ndarray[Shape[3, 3], Numeric]:
    '''
    Computes a linear transformation which, when applied to an arbitrary vector,
    yields a vector orthogonal to both that vector and the direction vector provided here
    Equivalent to taking the cross product of v with the direction vector
    
    Returns matrix which represents the orthogonalization transformation
    '''
    (dims,) = direction_vector.shape # implicitly enforce 1D shape for vector
    return np.cross(
        np.eye(dims, dtype=direction_vector.dtype),
        direction_vector / np.linalg.norm(direction_vector),
    ) 
cross = orthogonalizer

def rotator(direction_vector : np.ndarray[Shape[3], Numeric], angle_rad : float=0.0) -> np.ndarray[Shape[3, 3], Numeric]:
    # DEVNOTE: deprecate in favor of scipy.spatial..transform.Rotation.from_rotvec(), etc.
    ''' 
    Computes a linear transformation which, when applied to an arbitrary vector,
    rotates that vector by "angle_rad" radians around the axis defined by "direction_vector"
    (in a right-handed coordinate systems), as calculated by Rodrigues' rotation formula.
    
    Returns an orthogonal matrix which represents the rotation transformation. 
    '''
    (dims,) = direction_vector.shape # implicitly enforce 1D shape for vector
    I = np.eye(dims, dtype=direction_vector.dtype)
    K = orthogonalizer(direction_vector)
    
    return I + np.sin(angle_rad)*K + (1 - np.cos(angle_rad))*(K @ K)
rodrigues = rotator
