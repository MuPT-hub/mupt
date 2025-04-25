'''For handling array conversions to and from homogeneous coordinates'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional, Union
from ..arraytypes import Shape, N, M, Dims, DimsPlus, Numeric

import numpy as np


def to_homogeneous_coords(coords : np.ndarray[Shape[N, Dims], Numeric], projection : float=1.0) -> np.ndarray[Shape[N, DimsPlus], Numeric]:
    '''Convert an array of N points in D dimensions to one of N points in
    [D + 1] dimensions, with arbitrary uniform projection (1.0 by default)'''
    pad_widths = np.zeros((coords.ndim, 2), dtype=int) # pad nowhere...
    pad_widths[-1, -1] = 1 # ...EXCEPT exactly one layer AFTER the final dimension (and 0 before)
    
    return np.pad(coords, pad_width=pad_widths, mode='constant', constant_values=projection)
      
def from_homogeneous_coords(coords : np.ndarray[Shape[N, DimsPlus], Numeric]) -> np.ndarray[Shape[N, Dims], Numeric]:
    '''Project down from an array of N points in [D + 1] dimensions to an array
    of N points in D dimensions, normalizing out by the homogeneous coordinate'''
    return coords[..., :-1] / coords[..., -1, None] # strip off and normalize by the projective part (via broadcast)

def affine_matrix_from_linear_and_center(
        matrix : np.ndarray[Shape[Dims, Dims], Numeric],
        center : Optional[np.ndarray[Shape[Dims], Numeric]],
        dtype : Optional[type]=None,
    ) -> np.ndarray[Shape[DimsPlus, DimsPlus], Numeric]:
        '''Instantiate an affine transformation matrix from a linear transformation and a new origin location'''
        (n_rows, n_cols) = matrix.shape
        assert n_rows == n_cols
        dimension = n_cols
        
        if dtype is None:
            dtype = matrix.dtype
        
        if center is None:
            center = np.zeros(dimension, dtype=dtype)
        
        affine_matrix = np.zeros((dimension + 1, dimension + 1), dtype=dtype)
        affine_matrix[:-1, :-1] = matrix
        affine_matrix[:-1, -1]  = center
        affine_matrix[-1, -1]   = 1
        
        return affine_matrix

def apply_affine_transform_to_points(
        coords : np.ndarray[Shape[N, Dims], Numeric],
        transform : np.ndarray[Shape[DimsPlus, DimsPlus], Numeric],
        preserve_vector_shape : bool=True
    ):
    '''Take a vector of coordinates in D dimensions, apply a [D + 1] dimensional affine
    transformation, then project back down to D dimensions and return the output'''
    # TODO: check that matrix is compatible shape
    result = from_homogeneous_coords(to_homogeneous_coords(coords) @ transform.T)
    if preserve_vector_shape:
        result = np.squeeze(result) # squeeze preserves 1-dimensionality when applied to vectors

    return result