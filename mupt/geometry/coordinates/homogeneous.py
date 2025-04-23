'''For handling array conversions to and from homogeneous coordinates'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Union
from ..arraytypes import Shape, N, M, Dims, DimsPlus, Numeric

import numpy as np


def coordlike(coords : Union[np.ndarray[Shape[Dims], Numeric], np.ndarray[Shape[N, Dims], Numeric]]) -> np.ndarray[Shape[N, Dims], Numeric]:
    '''Standardizes input for applications which expect an NxD vector off D-dimensional coordinates
    Specifically handles the case of broadcasting 1-dimensional arrays into the right 2-dimensional shape'''
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    assert coords.ndim == 2
    
    return coords

def to_homogeneous_coords(coords : np.ndarray[Shape[N, Dims], Numeric], projection : float=1.0) -> np.ndarray[Shape[N, DimsPlus], Numeric]:
    '''Convert an array of N points in D dimensions to one of N points in
    [D + 1] dimensions, with arbitrary uniform projection (1.0 by default)'''
    coords = coordlike(coords)
    n_points, dimension = coords.shape
    
    return np.hstack([coords, np.full(shape=(n_points, 1), fill_value=projection, dtype=coords.dtype)])
    
def from_homogeneous_coords(coords : np.ndarray[Shape[N, DimsPlus], Numeric]) -> np.ndarray[Shape[N, Dims], Numeric]:
    '''Project down from an array of N points in [D + 1] dimensions to an array
    of N points in D dimensions, normalizing out by the homogeneous coordinate'''
    coords = coordlike(coords)
    
    return coords[:, :-1] / coords[:, [-1]]

def apply_affine_transform_to_points(coords : np.ndarray[Shape[N, Dims], Numeric], transform : np.ndarray[Shape[DimsPlus, DimsPlus], Numeric]):
    '''Take a vector of coordinates in D dimensions, apply a [D + 1] dimensional affine
    transformation, then project back down to D dimensions and return the output'''
    # TODO: check that matrix is compatible shape
    return from_homogeneous_coords(to_homogeneous_coords(coords) @ transform.T)