'''For determining and adjusting the sizes (measures) of geometric objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Optional, Union
import numpy as np

from .arraytypes import Shape, N, Numeric


def normalize(
        vector : np.ndarray[Shape[Any], Numeric],
        order  : Optional[Union[int, float, str]]=None,
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
        vector : np.ndarray[Shape[N, ...], Numeric],
        order  : Optional[Union[int, float, str]]=None,
    ) -> np.ndarray[Shape[N, ...], Numeric]:
    '''Return a normalized copy of a vector or array of vectors;
    The array supplied to "vector" is unchanged'''
    new_vector = np.copy(vector)  # preserve original vector
    normalize(new_vector, order=order)

    return new_vector