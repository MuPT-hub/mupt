'''Unit tests for homogeneous coordinate conversion and affine transforms'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest
from typing import Any
import numpy as np

from mupt.geometry.transforms.affine.homogeneous import (
    to_homogeneous_coords,
    from_homogeneous_coords,
)
from mupt.geometry.arraytypes import Shape, N, M, Dims, DimsPlus, Numeric


N : int = 10
point  = np.random.random((3,))
vector = np.random.random((N, 3))
block  = np.random.random((N, N, 3))

@pytest.mark.parametrize('array', (point, vector, block,))
def test_to_homogeneous_coords(array : np.ndarray[Shape[Any, 3], Numeric]) -> None:
    '''Test the conversion of arbitrarily-nested arrays of 3D coordinates to homogeneous coordinates'''
    projection : int = 1.0
    *array_shape, xyz_shape = array.shape
    expected_shape = tuple(array_shape + [xyz_shape + 1]) # note the first + is list concatenation, while the second is scalar addition
    
    homog = to_homogeneous_coords(array, projection=projection)
    assert (homog.shape == expected_shape) and np.allclose(homog[..., -1], projection)
    
@pytest.mark.parametrize('array', (point, vector, block,))
def test_from_homogeneous_coords(array : np.ndarray[Shape[Any, 3], Numeric]) -> None:
    '''Test that conversion to and back from homogeneous coordinates has no side effects'''
    piped_array = from_homogeneous_coords(to_homogeneous_coords(array))
    assert np.allclose(piped_array, array)