'''Calculations of local coordinate systems and linear bases'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from ..arraytypes import Shape, N, Dims, Numeric


def is_orthogonal(matrix : np.ndarray[Shape[N, N], Numeric]) -> bool:
    '''
    Determine if a matrix is orthogonal, i.e. its left and right inverses are both its own transpose
    Note that the matrix does not necessailry have to be square in order for it to be orthogonal
    '''
    (n_rows, n_cols) = matrix.shape # implicitly assert 2-dimensionality
    
    return  np.allclose(matrix @ matrix.T, np.eye(n_rows, dtype=matrix.dtype)) \
        and np.allclose(matrix.T @ matrix, np.eye(n_cols, dtype=matrix.dtype)) 

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