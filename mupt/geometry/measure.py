'''For determining and adjusting the sizes (measures) of geometric objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional, Union
import numpy as np

from .arraytypes import (
    N,
    NumericNP,
    OrderType,
    Shape,
    VectorN,
    ArrayNxM,
)


def normalize(
    vector : VectorN | ArrayNxN,
    order : OrderType=None,
) -> None:
    '''Normalize a vector or array of vectors in-place'''
    norms = np.atleast_1d( # ensure shape is broadcastable, even for scalars
        np.linalg.norm(vector, ord=order, axis=-1, keepdims=True)
    )
    ## DEVNOTE: thought about setting 0 entries in norm vector to 1's to avoid division by zero,
    ## but opted instead for clear Exception being raised by numpy when attempting division by zero
    # norms[np.isclose(norms, 0.0)] = 1.0  # avoid division by zero
    vector /= norms

def normalized(
    # DEV: using generic here to indicate return has same dtype
    vector : np.ndarray[Shape[N, ...], NumericNP], 
    order : OrderType=None,
) -> np.ndarray[Shape[N, ...], NumericNP]:
    '''
    Return a normalized copy of a vector or array of vectors;
    The array supplied to "vector" is unchanged
    '''
    new_vector = np.copy(vector)
    normalize(new_vector, order=order)

    return new_vector

def compare_optional_positions(
    position_1 : Optional[VectorN],
    position_2 : Optional[VectorN],
    radius : float=1E-8,
    order : OrderType=None,
) -> bool:
    '''Check that two vectors are within a certain absolute distance of one another'''
    # TODO: check vector shapes match
    if not (isinstance(position_1, np.ndarray) and isinstance(position_2, np.ndarray)):
        raise TypeError(f'Expected position attributes to be numpy.ndarray, got {type(position_1)} and {type(position_2)}')
    return (np.linalg.norm(position_1 - position_2, ord=2, axis=-1) < radius).astype(object) # cast to Python bool
