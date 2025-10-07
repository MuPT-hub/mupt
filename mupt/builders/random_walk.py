'''Utilities for generating coordinates of random walks'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generator, Optional, Union
from numbers import Number

import numpy as np
from ..geometry.arraytypes import Shape, Dims
from ..mutils.iteration import ad_infinitum


def random_direction(dimension : Dims=3) -> np.ndarray[Shape[Dims], float]:
    '''Generate a random N-dimensional unit vector'''
    direction = 2*np.random.rand(dimension) - 1 # generate random 3-vector in [-1, 1]
    return direction / np.linalg.norm(direction) # normalize to unit length

def random_walk_jointed_chain(
    n_steps_max: int,
    step_size : Union[Number, Generator[float, None, None]],
    clip_angle : float=np.pi/4,
    dimension : Dims=3,
    starting_point : Optional[np.ndarray[Shape[Dims], float]]= None,
) -> np.ndarray[Shape[int, Dims], float]:
    '''Generate a random walk in N-dimensional real space representing a freely-jointed chain 
    whose consecutive step directions are constrained to be within a cone of a given angle and
    whose step sizes are either uniform or given by a generator
    
    Parameters
    ----------
    n_steps_max : int
        Maximum number of steps in the random walk.
    step_size : float or Generator[float, None, None]
        Step size for each step, either as a float for uniform step size 
        or as a generator the size of each subsequent step
    clip_angle : float = pi/4
        Maximum angle allowed between directions of subsequent steps
        Smaller values will result in walks more along the "same" direction
        Angle should be passed in radians, as a value from [-pi, pi]
    starting_point : np.ndarray[Shape[dimension], float] = None
        Starting point of the random walk. If None, the walk starts at the origin.
        
    Returns
    -------
    np.ndarray[Shape[N + 1, dimension]], float]
        Array of N + 1 points visited by the walk, where N is the number of steps
        N is equal to n_steps_max or the number of step sizes,
        if the step size is given as a generator which is exhausted in fewer steps
    '''
    if not -np.pi <= clip_angle <= np.pi:
        raise ValueError("Clip_angle must be in the range [-pi, pi]")
    c_max = np.cos(clip_angle) # negative angles are allowed since cosine is even
    
    if starting_point is None:
        starting_point = np.zeros(dimension, dtype=float) # by default, sstart at the origin
    assert starting_point.shape == (dimension,), "Starting point must be a 3D vector"
    
    if isinstance(step_size, Number):
        step_sizes = ad_infinitum(step_size)
    elif isinstance(step_size, Generator):
        step_sizes = step_size
    else: raise TypeError('step_size must be a float or a generator')
        
    steps = np.zeros((n_steps_max, dimension), dtype=float)
    step_direction_prev = None
    for i in range(n_steps_max):
        step_direction = random_direction(dimension=dimension)
        if step_direction_prev is None:
            step_direction_prev = step_direction
            
        # over [0, pi], cosine is monotonically decreasing, so overly-large steps will have cosine BELOW the cutoff
        while np.dot(step_direction, step_direction_prev) < c_max: 
            step_direction = random_direction(dimension=dimension) # draw new step within cone of movement by rejection sampling
        
        try:
            steps[i] = next(step_sizes)*step_direction
            step_direction_prev = step_direction # update previous step for next iteration
        except StopIteration:
            steps = steps[:i] # truncate steps before halting
            break

    return np.vstack([starting_point, starting_point + np.cumsum(steps, axis=0)]) # accumulate net travel for points
