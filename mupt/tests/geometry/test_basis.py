'''Unit tests for basis coordinate operations'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest
import numpy as np

from mupt.geometry.arraytypes import Shape, N
from mupt.geometry.coordinates.basis import (
    is_rowspace_mutually_orthogonal,
    is_columnspace_mutually_orthogonal,
    is_orthonormal,
)


# Test matrices for orthogonality checks
ORTHO_ROWS_ONLY = np.array([
    [ 1,  2,  3],
    [ 2,  2, -2],
    [ 5, -4,  1]
]) # an example of a matrix whose rows are mutually orthogonal but whose columns are not
ORTHO_COLS_ONLY = np.array([
    [ 1,  2,  5],
    [ 2,  2, -4],
    [ 3, -2,  1]
]) # an example of a matrix whose columns are mutually orthogonal but whose rows are not
RECTANGULAR = np.array([
    [1, -3, 0, 2],
    [4, 0, 3, -2],
]) # rowspace and columnspace checks should also work for nonsquare matrices - this one has orthogonal rows, but not columns
IDENTITY = np.eye(3) # identity matrix is orthogonal by definition
ROTATION = np.array([
    [np.cos(np.pi/3), -np.sin(np.pi/3), 0],
    [np.sin(np.pi/3),  np.cos(np.pi/3), 0],
    [0,                0,               1],
]) # by definition, proper rotation matrices are orthonormal


@pytest.mark.parametrize(
    'matrix, expected_value',
    [
        (ORTHO_ROWS_ONLY, True),
        (ORTHO_COLS_ONLY, False),
        (RECTANGULAR, True),
        (ROTATION, True),
        (IDENTITY, True),
    ]
)
def test_rowspace_orthogonality_check(matrix : np.ndarray[Shape[N, N], float], expected_value : bool) -> None:
    '''Test that the row space orthogonality check works as expected'''
    assert is_rowspace_mutually_orthogonal(matrix) == expected_value
    
@pytest.mark.parametrize(
    'matrix, expected_value',
    [
        (ORTHO_ROWS_ONLY, False),
        (ORTHO_COLS_ONLY, True),
        (RECTANGULAR, False),
        (ROTATION, True),
        (IDENTITY, True),
    ]
)
def test_columnspace_orthogonality_check(matrix : np.ndarray[Shape[N, N], float], expected_value : bool) -> None:
    '''Test that the column space orthogonality check works as expected'''
    assert is_columnspace_mutually_orthogonal(matrix) == expected_value

@pytest.mark.parametrize(
    'matrix, expected_value',
    [
        (ORTHO_ROWS_ONLY, False),
        (ORTHO_COLS_ONLY, False),
        (RECTANGULAR, False),
        (ROTATION, True),
        (IDENTITY, True),
    ]
)
def test_orthonormality_check(matrix : np.ndarray[Shape[N, N], float], expected_value : bool) -> None:
    '''Test that the orthonormality check works as expected'''
    assert is_orthonormal(matrix) == expected_value