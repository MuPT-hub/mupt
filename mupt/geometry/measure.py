'''For determining and adjusting the sizes (measures) of geometric objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional
import numpy as np

from .arraytypes import (
    N,
    NumericNP,
    OrderType,
    Shape,
    VectorN,
    ArrayNxM,
)
OrderType = Union[int, float, str] # types accepted by 'order' arg of np.linalg.norm()


def normalize(
    vector : VectorN | ArrayNxN,
    order : Optional[OrderType]=None,
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
    vector : np.ndarray[Shape[N, ...], NumericNP], # DEV: using generic here to indicate return has same dtype
    order  : Optional[OrderType]=None,
) -> np.ndarray[Shape[N, ...], NumericNP]:
    '''
    Return a normalized copy of a vector or array of vectors;
    The array supplied to "vector" is unchanged
    '''
    new_vector = np.copy(vector)  # preserve original vector
    normalize(new_vector, order=order)

    return new_vector

def compare_optional_positions(
    position_1 : Optional[VectorN],
    position_2 : Optional[VectorN],
    radius : float=1E-8,
    order : Optional[OrderType]=None,
) -> bool:
    '''
    Check that two positional values are either:
    * Both undefined (returns True)
    * Both defined AND within a set in distance in a given p-norm (returns True if both conditions are met)
    * One defined and one undefined, in either order (returns False)
    '''
    if type(position_1) != type(position_2):
        return False
    
    if position_1 is None: # both are None
        return True
    elif isinstance(position_1, np.ndarray):
        return (
            np.linalg.norm(
                position_1 - position_2,
                ord=order,
                axis=-1,
            ) < radius
        ).astype(object) # cast to Python bool
    else:
        raise TypeError(
            f'Expected positions to be either NoneType or numpy.ndarray: ' \
            f'got {type(position_1).__name__} and {type(position_2).__name__}'
        )
    