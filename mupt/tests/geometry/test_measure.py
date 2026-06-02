'''Unit tests for vector measure operations'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest

import numpy as np

from mupt.geometry.measure import normalize, normalized
from mupt.geometry.arraytypes import Shape, N, M


@pytest.mark.parametrize(
    'vector',
    [
        # TODO: test scalar
        # TODO: test N-vector
        # TODO: test 1xN vector    
        # TODO: test Nx1 vector    
        # TODO: test 2D array of vectors
    ],
)
def test_normalize(vector : np.ndarray[Shape[N, M], float]) -> np.ndarray[Shape[N, M]]:
    ...
    