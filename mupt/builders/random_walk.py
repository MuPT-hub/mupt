'''Utilities for generating coordinates of random walks'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import Generator, Iterable, Optional, Sized, Union
from numbers import Number

import numpy as np

from .base import PlacementGenerator
from ..mutils.iteration import flexible_iterator

from ..geometry.arraytypes import Shape, Dims
from ..geometry.shapes import Ellipsoid, Sphere
from ..geometry.measure import normalized
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.coordinates.reference import CoordAxis, origin


def random_walk_jointed_chain(
    step_size : Union[Number, Iterable[Number], Generator[Number, None, None]],
    n_steps_max : Optional[int]=None,
    initial_point : Optional[np.ndarray[Shape[Dims], float]]=None,
    initial_direction : Optional[np.ndarray[Shape[Dims], float]]=None,
    clip_angle : float=np.pi/4,
    dimension : Dims=3,
) -> Generator[np.ndarray[Shape[Dims], float], None, None]:
    '''
    Generate consecutive points from a non-self-avoiding random walk in continuous N-dimensional space
    with arbitrary step sizes that are constrained within a prescribed angle between consecutive steps

    Parameters
    ----------
    step_size : float or Generator[float, None, None]
        Step size for each step, either as a float for uniform step size 
        or as a generator the size of each subsequent step
    initial_point : np.ndarray[Shape[dimension], float] = None
        Starting point of the random walk. If None, the walk starts at the origin.
    initial_direction : np.ndarray[Shape[dimension], float] = None
        Initial direction of the first step. If None, a random direction is chosen.
    n_steps_max : int
        Maximum number of steps in the random walk.
    clip_angle : float = pi/4
        Maximum angle allowed between directions of subsequent steps
        Smaller values will result in walks more along the "same" direction
        Angle should be passed in radians, as a value from [-pi, pi]
    dimension : int, default 3
        Dimension of the space in which the random walk is performed
        If no start point is provided, the inferred origin used as the start will have this many dimensions
        
    Returns
    -------
    Generator[np.ndarray[Shape[dimension], float], None, None]
        A generator yielding conseuctive positions along the walk, starting with initial_position
    '''
    # validate preconditions
    if not -np.pi <= clip_angle <= np.pi:  # negative angles are allowed since cosine is even
        raise ValueError("Clip_angle must be in the range [-pi, pi]")
    c_max = np.cos(clip_angle)
    
    if initial_point is None:
        initial_point = origin(dimension=dimension)
    if initial_point.shape != (dimension,): # NOTE: check user-provided start shape or (redundantly-but-safely) the shape of the auto-assigned start)
        raise ValueError(f"Random walk starting point must be a {dimension}-dimensional vector")
    
    if initial_direction is None:
        initial_direction = random_unit_vector(dimension=dimension)
    assert initial_direction.shape == (dimension,) # NOTE: check user-provided start direction shape

    if (n_steps_max is None):
        if not isinstance(step_size, Iterable):
            LOGGER.warning('No upper bound supplied for number of random walk steps; singly-valued step size will produce steps indefinitely!')
        elif not isinstance(step_size, Sized):
            LOGGER.warning('No upper bound supplied for number of random walk steps; unbounded step size iterator *MAY* produce steps indefinitely!')
    
    # generate walk points
    n_steps_taken : int = 0
    net_position = initial_point
    prev_direction = normalized(initial_direction)
    
    yield initial_point # doesn't count as a step yet
    for step_size in flexible_iterator(step_size, allowed_types=(Number,)):
        # draw new step within cone of movement by rejection sampling (simple and quick)
        step_direction = random_unit_vector(dimension=dimension)
        while np.dot(step_direction, prev_direction) < c_max: # over [0, pi], cosine is monotonically decreasing, so overly-large steps will have cosine BELOW the cutoff
            step_direction = random_unit_vector(dimension=dimension)
        net_position += (step_size * step_direction)
        
        yield net_position

        prev_direction = step_direction
        n_steps_taken += 1 # only increment AFTER we've actually yielded
        if (n_steps_max is not None) and (n_steps_taken >= n_steps_max):
            break


class AngleConstrainedRandomWalk(PlacementGenerator):
    '''
    Simple demonstration builder which places children of a Primitive
    in a non-self-avoiding, constrained-angle random walk
    '''
    def __init__(self, angle : float=np.pi/4) -> None:
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