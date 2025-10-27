'''For procedurally generating random coordinate directions'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np

from .basis import are_linearly_independent
from ..arraytypes import Shape, Dims, Numeric
from ..measure import normalize
from ..transforms.linear import rejector


def random_vector(
    dimension : Dims=3,
    low : float=-1.0,
    high : float=1.0,
    normalized : bool=False,
) -> np.ndarray[Shape[Dims], float]:
    '''Generate a random N-dimensional vector of floats, optionally normalized'''
    vector = np.random.uniform(low=low, high=high, size=dimension)
    if normalized:
        normalize(vector)
        
    return vector

def random_unit_vector(dimension : Dims=3) -> np.ndarray[Shape[Dims], float]:
    '''Generate a randomly-oriented unit vector in N-dimensional space'''
    return random_vector(dimension=dimension, low=-1.0, high=1.0, normalized=True)

def random_orthogonal_vector(
    vector : np.ndarray[Shape[Dims], Numeric],
    normalized : bool=True,
) -> np.ndarray[Shape[Dims], Numeric]:
    '''Return a random vector orthogonal to the input vector'''
    (ndim,) = vector.shape
    if np.allclose(vector, 0.0):
        raise ValueError('No vector can be orthogonal to the zero vector')
    
    random_direction = np.copy(vector)
    while not are_linearly_independent(random_direction, vector): # rejection sampling avoids colinear vector edge case
        random_direction = random_vector(ndim, low=-1.0, high=1.0, normalized=True)

    tangent = rejector(vector) @ random_direction
    if normalized:
        normalize(tangent)
    assert np.isclose(np.dot(tangent, vector), 0.0), 'Calculation error; tangent vector not orthogonal to input vector'
        
    return tangent
random_tangent = random_orthogonal_vector  # alias for convenience