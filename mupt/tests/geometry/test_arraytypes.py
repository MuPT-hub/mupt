'''Unit tests for array size and dtype typehinting'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest
import numpy as np

from mupt.geometry.arraytypes import (
    as_n_vector,
    VectorN,
)


@pytest.fixture
def vector_expected() -> VectorN:
    return np.array([1.,2.,3.])

@pytest.mark.parametrize(
    'vectorlike',
    [
        ([1,2,3]),
        [[1,2,3]],
        tuple([1,2,3]),
        np.array([1,2,3]),
        np.array([1.,2.,3.]),
        np.array([[1,2,3]]),
        np.array([[1,2,3]]).T,
        pytest.param(
            [1,2,3,4],
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason='Input vectorlike has the wrong shape',
                strict=True,
            )
        ),
        pytest.param(
            '[1,2,3]',
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason='String of numerics is not a valid Sequence of numerics for interpretation as a vector',
                strict=True,
            )
        ),
    ],
)
def test_as_n_vector_shape(vectorlike, vector_expected : VectorN) -> None:
    '''Test that permissive vector ingestion accepts the kinds of numeric data structures it advertises'''
    vector_actual = as_n_vector(vectorlike, dimension=None) # suppress internal shape validation
    assert (vector_actual.size == (vector_expected.size))
