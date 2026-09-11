"""Unit tests for vector measure operations"""

import pytest

import numpy as np

from mupt.geometry.arraytypes import ArrayMxN


@pytest.mark.parametrize(
    "vector",
    [
        # TODO: test scalar
        # TODO: test N-vector
        # TODO: test 1xN vector
        # TODO: test Nx1 vector
        # TODO: test 2D array of vectors
    ],
)
def test_normalize(vector: ArrayMxN) -> ArrayMxN:
    """
    Test that normalize() properly normalizes
    various vector and array-like objects
    """
    ...
