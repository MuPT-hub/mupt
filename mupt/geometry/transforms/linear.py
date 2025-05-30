'''Application and construction of common linear transformations of points in 3D space'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from scipy.spatial.transform import Rotation

from ..arraytypes import Shape, N, Dims, Numeric


def parallel_projector(direction_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector parallel to the direction vector provided
    
    Returns matrix which represents the parallel projector transformation'''
    (dim,) = direction_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    return np.outer(direction_vector, direction_vector) / np.inner(direction_vector, direction_vector)

def orthogonal_projector(direction_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector orthogonal to the direction vector provided
    
    Returns matrix which represents the orthogonal projector transformation'''
    (dim,) = direction_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    return np.eye(dim, dtype=direction_vector.dtype) - parallel_projector(direction_vector) # equivalent to substracting parallel part off of vector transform is applied to
perpendicular_projector = orthogonal_projector # alias for convenience

# TODO: make separate constructor function for Householder matrices

def alignment_transform(initial_vector : np.ndarray[Shape[Dims], Numeric], final_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''Compute an orthogonal linear transformation which takes a given initial vector to a final vector
    while preserving the relative orientations of the basis vectors to one another
    
    Returns an orthogonal Householder matrix which represents the transformation'''
    (dim,) = initial_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    assert final_vector.shape == initial_vector.shape # ensure both vectors have 
    diff = final_vector - initial_vector # could arbitrarily reverse order, will only differ in sign of reflection plane normal
    
    return np.eye(dim, dtype=initial_vector.dtype) - 2*parallel_projector(diff) # compute Householder reflection
