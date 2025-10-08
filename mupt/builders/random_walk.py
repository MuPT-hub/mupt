'''Utilities for generating coordinates of random walks'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generator, Optional, Union, TypeVar, Iterable, Iterator
from numbers import Number

import numpy as np

from .base import PlacementGenerator
from ..mutils.iteration import flexible_iterator

from ..geometry.arraytypes import Shape, Dims
from ..geometry.shapes import Ellipsoid, Sphere
from ..geometry.coordinates.directions import random_unit_vector


def random_walk_jointed_chain(
    n_steps_max: int,
    step_size : Union[Number, Iterable[Number], Generator[Number, None, None]],
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
    
    step_sizes = flexible_iterator(step_size, allowed_types=(Number,))
    steps = np.zeros((n_steps_max, dimension), dtype=float)
    step_direction_prev = None
    for i in range(n_steps_max):
        step_direction = random_unit_vector(dimension=dimension)
        if step_direction_prev is None:
            step_direction_prev = step_direction
            
        # over [0, pi], cosine is monotonically decreasing, so overly-large steps will have cosine BELOW the cutoff
        while np.dot(step_direction, step_direction_prev) < c_max: 
            step_direction = random_unit_vector(dimension=dimension) # draw new step within cone of movement by rejection sampling
        
        try:
            steps[i] = next(step_sizes)*step_direction
            step_direction_prev = step_direction # update previous step for next iteration
        except StopIteration:
            steps = steps[:i] # truncate steps before halting
            break

    return np.vstack([starting_point, starting_point + np.cumsum(steps, axis=0)]) # accumulate net travel for points


class RandomWalkDemoBuilder(CoordinateBuilder):
    '''
    Simple demonstration builder which places children of a Primitive
    in a non-self-avoiding, constrained-angle random walk
    '''
    def __init__(self, angle : float) -> None:
        ...
    
    def check_preconditions(self, primitive):
        if primitive.topology.is_branched:
            raise ValueError('Random walk chain builder behavior undefined for branched topologies')
        
        if not all(isinstance(subprim.shape, (Sphere, Ellipsoid)) for subprim in primitive.children):
            raise TypeError('Random walk chain builder requires ellipsoidal of spherical beads to determine step sizes')
    
    def _generate_coordinates(self, primitive):
        for chain in primitive.topology.chains:
            termini = tuple(chain.termini)
            if len(termini) == 1: # DEV: opted for more readable impl here; iter-based maybe more efficient, but really doesn't save much
                head_grp, tail_grp = termini[0], termini[0]
            elif len(termini) == 2:
                head_grp, tail_grp = termini
            else:
                raise ValueError('Unbranched topology must have either 1 or 2 terminal nodes')
            
            
        
        ## locate start of each chain
        
        # for all chains (connected componenets):
        ## determine traversal order 
        ## determine step size per bead
        
        ## sppol off orients from RW