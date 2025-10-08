'''Utilities for generating coordinates of random walks'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import (
    Generator,
    Hashable,
    Iterable,
    Optional,
    Sized,
    Union,
)
from numbers import Number
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import RigidTransform
from networkx import all_simple_paths

from .base import PlacementGenerator
from ..mutils.iteration import flexible_iterator, sliding_window

from ..geometry.arraytypes import Shape, Dims
from ..geometry.measure import normalized
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.coordinates.reference import origin
from mupt.geometry.transforms.rigid import rigid_vector_coalignment

from ..mupr.topology import TopologicalStructure
from ..mupr.connection import Connector
from ..mupr.primitives import Primitive, PrimitiveHandle


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
    initial_point : ndarray[Shape[dimension], float] = None
        Starting point of the random walk. If None, the walk starts at the origin.
    initial_direction : ndarray[Shape[dimension], float] = None
        Initial direction of the first step. If None, a random direction is chosen.
    n_steps_max : int
        Maximum number of steps in the random walk
        Will result in (n_steps_max + 1) points being generated (+1 to include the initial point)
    clip_angle : float = pi/4
        Maximum angle allowed between directions of subsequent steps
        Smaller values will result in walks more along the "same" direction
        Angle should be passed in radians, as a value from [-pi, pi]
    dimension : int, default 3
        Dimension of the space in which the random walk is performed
        If no start point is provided, the inferred origin used as the start will have this many dimensions
        
    Returns
    -------
    Generator[ndarray[Shape[dimension], float], None, None]
        A generator yielding conseuctive positions along the walk, starting with initial_position
    '''
    # validate preconditions
    if not -np.pi <= clip_angle <= np.pi:  # negative angles are allowed since cosine is even
        raise ValueError("Clip_angle must be in the range [-pi, pi]")
    cos_max : float = np.cos(clip_angle)
    
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
    net_position : np.ndarray = initial_point
    prev_direction : np.ndarray = normalized(initial_direction)
    
    yield initial_point # always yielded, consider as "step #0"
    for step_size in flexible_iterator(step_size, allowed_types=(Number,)):
        # draw new step within cone of movement by rejection sampling (simple and quick)
        step_direction : np.ndarray = random_unit_vector(dimension=dimension)
        while np.dot(step_direction, prev_direction) < cos_max: # NOTE: over |x| in [0, pi], cos(x) is monotonically decreasing, so overly-large steps will have cosine BELOW the cutoff
            step_direction : np.ndarray = random_unit_vector(dimension=dimension)
        net_position += (step_size * step_direction)
        
        yield net_position

        prev_direction : np.ndarray = step_direction
        n_steps_taken += 1 # only increment AFTER we've actually yielded
        if (n_steps_max is not None) and (n_steps_taken >= n_steps_max):
            break

class AngleConstrainedRandomWalk(PlacementGenerator):
    '''
    Simple demonstration builder which places children of a Primitive
    in a non-self-avoiding, constrained-angle random walk
    '''
    def __init__(
        self,
        angle_max_rad : float=np.pi/4,
        initial_point : Optional[np.ndarray[Shape[3], float]]=None,
        initial_direction : Optional[np.ndarray[Shape[3], float]]=None,
    ) -> None:
        self.angle_max_rad = angle_max_rad
        self.initial_point = initial_point
        self.initial_direction = initial_direction

    # optional helper methods (to declutter casework from main logic)
    def get_termini_handles(self, chain : TopologicalStructure) -> tuple[Hashable, Hashable]:
        '''
        Find the terminal node(s) of what is assumed to be a linear (path) graph
        Returns the pair of node labels of the termini (a pair of the same value twice for single-node graphs)
        '''
        termini = tuple(chain.termini)
        if len(termini) == 2:
            return termini
        elif len(termini) == 1: 
            return termini[0], termini[0]
        else:
            raise ValueError('Unbranched topology must have either 1 or 2 terminal nodes')

    # implementing builder contracts
    def check_preconditions(self, primitive : Primitive) -> None:
        '''Enforce that no branches chains exist anywhere'''
        if primitive.topology.is_branched:
            raise ValueError('Random walk chain builder behavior undefined for branched topologies')
        
        if any((subprim.shape is None) for subprim in primitive.children):
            raise TypeError('Random walk chain builder requires ellipsoidal of spherical beads to determine step sizes')
    
    def _generate_placements(self, primitive : Primitive) -> Generator[tuple[PrimitiveHandle, RigidTransform], None, None]:
        '''
        Reorient bodies to be coincident (along a predefined axis) with 
        the steps of an angle-constrained non-self-avoiding random walk
        '''
        for chain in primitive.topology.chains:
            # DEV: taking extra care to ensure chain is oriented from end-to-end, because there's no requirement
            # (or indeed, reason to believe) that the order of nodes in chain.nodes is meaningful
            head_handle, tail_handle = termini = self.get_termini_handles(chain)
            path : list[PrimitiveHandle] = next(all_simple_paths(chain, source=head_handle, target=tail_handle)) # raise StopIteration if no path exists
            
            # determine pair of anchor points per-body that alignment is based upon
            connection_points : dict[PrimitiveHandle, list[np.ndarray, np.ndarray]] = defaultdict(list)
            connection_points[head_handle].append(primitive.children_by_handles[head_handle].shape.centroid)
            for prim_handle_outgoing, prim_handle_incoming in sliding_window(path, 2):
                conn_handle_outgoing, conn_handle_incoming = primitive.topology.internal_connections_between(
                    from_child_handle=prim_handle_outgoing,
                    to_child_handle=prim_handle_incoming,
                )
                # NOTE: traversal in-path-order is what guarantees these appends place everything in the correct order
                conn_outgoing = primitive.fetch_connector_on_child(prim_handle_outgoing, conn_handle_outgoing)
                connection_points[prim_handle_outgoing].append(conn_outgoing.anchor_position) # will raise Exception is anchor position is unset
                
                conn_incoming = primitive.fetch_connector_on_child(prim_handle_incoming, conn_handle_incoming)
                connection_points[prim_handle_outgoing].append(conn_incoming.anchor_position) # will raise Exception is anchor position is unset
                
                Connector.mutually_antialign_ballistically(conn_outgoing, conn_incoming) # align linkers w/ other's anchor while leaving anchors themselves undisturbed
            # NOTE: order is critical here; only placing tail point AFTER its incoming connection point is inserted
            connection_points[tail_handle].append(primitive.children_by_handles[tail_handle].shape.centroid)
            
            # generate random walk steps and corresponding placements
            rw_steps : Generator[np.ndarray, None, None] = random_walk_jointed_chain(
                step_size=(self.determine_step_size(primitive.children[handle].shape) for handle in path),
                n_steps=len(path), # not strictly necessary, but suppresses "indeterminate num steps" warnings
                initial_point=self.initial_point,
                initial_direction=self.initial_direction,
                clip_angle=self.angle_max_rad,
                dimension=3,
            )
            for handle, (step_start, step_end) in zip(path, sliding_window(rw_steps, 2)):
                conn_start, conn_end = connection_points[handle]
                t_body = 0.0 if handle in termini else 0.5 # terminal Primitives have second point set to center, not middle
                
                placement_transform = rigid_vector_coalignment(
                    # vector 1: spans between anchors of connection point on body
                    conn_start,
                    conn_end,
                    # vector 2: spans between consecutive random walk steps
                    step_start,
                    step_end,
                    # interpolation parameters for which point on respective vectors will be forced exactly-coexistent
                    t1=t_body, # take midpoint (or end, if at termini) of body-anchoring vector
                    t2=0.5, # ...to midpoint of random walk step vector
                )
                yield handle, placement_transform
