"""Unit tests for homogeneous coordinate conversion and affine transforms"""

import pytest
import numpy as np

from mupt.geometry.transforms.affine.homogeneous import (
    to_homogeneous_coords,
    from_homogeneous_coords,
)
from mupt.geometry.arraytypes import Vector3


N: int = 10


def tensor_examples(N: int = 10) -> tuple[np.ndarray, ...]:
    """
    Various tensors with random numerical entries but consistent dimension
    Used to test that results of homogeneous coordinate
    conversion operations have the excepted shapes
    """
    point = np.random.random((3,))
    vector = np.random.random((N, 3))
    block = np.random.random((N, N, 3))

    return (point, vector, block)


@pytest.mark.parametrize("array", tensor_examples(10))
def test_to_homogeneous_coords(array: Vector3) -> None:
    """
    Test the conversion of arbitrarily-nested arrays
    of 3D coordinates to homogeneous coordinates
    """
    projection: float = 1.0
    *array_shape, xyz_shape = array.shape
    expected_shape = tuple(
        array_shape + [xyz_shape + 1]
    )  # note the first + is list concatenation, while the second is scalar addition

    homog = to_homogeneous_coords(array, projection=projection)
    assert (homog.shape == expected_shape) and np.allclose(homog[..., -1], projection)


@pytest.mark.parametrize("array", tensor_examples(10))
def test_from_homogeneous_coords(array: Vector3) -> None:
    """
    Test that conversion to and back from
    homogeneous coordinates has no side effects
    """
    piped_array = from_homogeneous_coords(to_homogeneous_coords(array))
    assert np.allclose(piped_array, array)
