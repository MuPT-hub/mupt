'''Calculation and application of common linear and affine transformations'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional, TypeVar
T = TypeVar('T')

import numpy as np
from scipy.spatial.transform import Rotation

from ..arraytypes import Shape, N, Dims


# Projection and Reflection
def parallel_projector(direction_vector : np.ndarray[Shape[Dims], T]) -> np.ndarray[Shape[Dims, Dims], T]:
    '''Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector parallel to the direction vector provided
    
    Returns matrix which represents the parallel projector transformation'''
    (dim,) = direction_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    return np.outer(direction_vector, direction_vector) / np.inner(direction_vector, direction_vector)

def orthogonal_projector(direction_vector : np.ndarray[Shape[Dims], T]) -> np.ndarray[Shape[Dims, Dims], T]:
    '''Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector orthogonal to the direction vector provided
    
    Returns matrix which represents the orthogonal projector transformation'''
    (dim,) = direction_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    return np.eye(dim, dtype=direction_vector.dtype) - parallel_projector(direction_vector) # equivalent to substracting parallel part off of vector transform is applied to
perpendicular_projector = orthogonal_projector # alias for convenience

def alignment_transform(initial_vector : np.ndarray[Shape[Dims], T], final_vector : np.ndarray[Shape[Dims], T]) -> np.ndarray[Shape[Dims, Dims], T]:
    '''Compute an orthogonal linear transformation which takes a given initial vector to a final vector
    while preserving the relative orientations of the basis vectors to one another
    
    Returns an orthogonal Householder matrix which represents the transformation'''
    (dim,) = initial_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    assert final_vector.shape == initial_vector.shape # ensure both vectors have 
    diff = final_vector - initial_vector # could arbitrarily reverse order, will only differ in sign of reflection plane normal
    
    return np.eye(dim, dtype=initial_vector.dtype) - 2*parallel_projector(diff) # compute Householder reflection


# SVD and Local coordinates
def compute_local_coordinates(positions : np.ndarray[Shape[N, Dims], float]) -> tuple[
        np.ndarray[Shape[Dims], float],
        np.ndarray[Shape[Dims, Dims], float],
        np.ndarray[Shape[Dims], float],
    ]:
    '''
    Takes a coordinates vector of N D-dimensional points and determines
    the center, axes, and relative lengths of the local principal coordinate systems
    
    Parameters
    ----------
    positions : Array[[N, D], float]
        A vector of N points in D-dimensional
    
    Returns
    -------
    center : Array[[D,], float]
        The D-dimensional coordinate point of the local coordinate origin
    principal_axes : Array[[D, D], float]
        A DxD matrix whose i-th column is the i-th basis vector in the local coordinate system
        Basis provided is orthonormal (i.e. all columns have length 1 and are perpendicular to each other column)
    axis_lengths : Array[[D,], float]
        The relative length of each axis, if ordered by significance (i.e. amount of variation along that axis)
    '''
    center = positions.mean(axis=0)
    
    # determine principal axes from SVD
    U, S, Vh = np.linalg.svd((positions - center), full_matrices=False) # NOTE: this places eigenvalues in descending order by default (no sorting needed)
    principal_axes = eivecs = Vh.T          # transpose to place eigenvectors into column-order - NOTE: basis is guaranteed to be normal, since covariance matrix is real and symmetric
    axis_lengths = eivals = (S * S) / (len(positions) - 1) # account for sample size normalization for covariance matrix
    
    return center, principal_axes, axis_lengths