'''Calculations to determine if linear bases enjoy certian properties, such as orthogonality'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from ..arraytypes import Shape, N, Numeric


def are_linearly_independent(*vectors: np.ndarray[Shape[N, ...], Numeric]) -> bool:
    '''Check if the given set of vectors is linearly independent'''
    # NOTE: numpy will raise Exception (as desired) when vectors passed have incompatible shapes
    return np.linalg.matrix_rank(np.column_stack(vectors)) == len(vectors)

def is_diagonal(matrix : np.ndarray[Shape[N, N], Numeric]) -> bool: # TODO: generalize to work for other diagonals
    '''Determine whether a matrix is digonal, i.e. has no nonzero elements off of the main diagonal'''
    return np.allclose(matrix - np.diag(np.diagonal(matrix)), 0.0)

def is_rowspace_mutually_orthogonal(matrix : np.ndarray[Shape[N, N], Numeric]) -> bool:
    '''Check whether all vectors in the row space basis of a matrix are mutually orthogonal'''
    return is_diagonal(matrix @ matrix.T) # note CAREFULLY the order; P_ij = dot(row(i), row(j)) this way

def is_columnspace_mutually_orthogonal(matrix : np.ndarray[Shape[N, N], Numeric]) -> bool:
    '''Check whether all vectors in the column space basis of a matrix are mutually orthogonal'''
    return is_diagonal(matrix.T @ matrix) # note CAREFULLY the order; P_ij = dot(column(i), column(j)) this way

def is_orthogonal(matrix : np.ndarray[Shape[N, N], Numeric]) -> bool:
    '''
    Determine if a matrix is orthogonal, i.e. its left and right inverses are both its own transpose
    Note that the matrix does not necessarily have to be square in order for it to be orthogonal
    '''
    (n_rows, n_cols) = matrix.shape # implicitly assert 2-dimensionality
    return  np.allclose(matrix @ matrix.T, np.eye(n_rows, dtype=matrix.dtype)) \
        and np.allclose(matrix.T @ matrix, np.eye(n_cols, dtype=matrix.dtype)) # NOTE: can't optimize as the transpose of the above product for non-square matrices
is_orthonormal = is_orthogonal
