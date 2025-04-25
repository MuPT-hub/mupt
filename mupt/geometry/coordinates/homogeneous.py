'''For handling array conversions to and from homogeneous coordinates'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Optional, Union
from ..arraytypes import Shape, N, M, Dims, DimsPlus, Numeric

import numpy as np


def to_homogeneous_coords(positions : np.ndarray[Shape[N, Dims], Numeric], projection : float=1.0) -> np.ndarray[Shape[N, DimsPlus], Numeric]:
    '''Convert an array of N points in D dimensions to one of N points in
    [D + 1] dimensions, with arbitrary uniform projection (1.0 by default)'''
    pad_widths = np.zeros((positions.ndim, 2), dtype=int) # pad nowhere...
    pad_widths[-1, -1] = 1 # ...EXCEPT exactly one layer AFTER the final dimension (and 0 before)
    
    return np.pad(positions, pad_width=pad_widths, mode='constant', constant_values=projection)
      
def from_homogeneous_coords(positions : np.ndarray[Shape[N, DimsPlus], Numeric]) -> np.ndarray[Shape[N, Dims], Numeric]:
    '''Project down from an array of N points in [D + 1] dimensions to an array
    of N points in D dimensions, normalizing out by the homogeneous coordinate'''
    return positions[..., :-1] / positions[..., -1, None] # strip off and normalize by the projective part (via broadcast)

def affine_matrix_from_linear_and_center(
        matrix : np.ndarray[Shape[Dims, Dims], Numeric],
        center : Optional[np.ndarray[Shape[Dims], Numeric]],
        dtype : Optional[type]=None,
    ) -> np.ndarray[Shape[DimsPlus, DimsPlus], Numeric]:
        '''
        Instantiate an affine transformation matrix from a linear transformation and a new origin location
        
        Parameters
        ----------
        matrix : Array[[D, D], Numeric]
            A D-dimensional linear transformation matrix
        center : Array[[D,], Numeric]
            A D-dimensional vector representing the new origin location
        dtype : type
            The data type of the output matrix
            If None, will be the same as the input matrix
            
        Returns
        -------
        Array[[D + 1, D + 1], Numeric]
            The corresponding [D + 1] dimensional affine transformation matrix
        '''
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
        positions : np.ndarray[Shape[Any, Dims], Numeric],
        transform : np.ndarray[Shape[DimsPlus, DimsPlus], Numeric],
    ) -> np.ndarray[Shape[Any, Dims], Numeric]:
    '''
    Take a vector of coordinates in D dimensions, apply a [D + 1] dimensional affine
    transformation, then project back down to D dimensions and return the output
    
    Parameters
    ----------
    positions : Array[[..., D], Numeric]
        An array of points in D-dimensional space
        Can be nested to any level of depth, as long as the last dimension is D
    transform : Array[[D + 1, D + 1], Numeric]
        A [D + 1] dimensional affine transformation matrix      
        
    Returns
    -------
    Array[[..., D], Numeric]
        An array of the transformed points in D-dimensional space
        Has the same shape as the input
    '''
    # TODO: check that matrix is compatible shape
    return from_homogeneous_coords(to_homogeneous_coords(positions) @ transform.T)