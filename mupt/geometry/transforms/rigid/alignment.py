'''For calculation of rigid transforms which force two bodies to be spatially coincident in some way'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from scipy.spatial.transform import RigidTransform

from .rotations import alignment_rotation
from ...arraytypes import Shape, N, Numeric


def rigid_vector_coalignment(
    vector1_start : np.ndarray[Shape[3], Numeric],
    vector1_end : np.ndarray[Shape[3], Numeric],
    vector2_start : np.ndarray[Shape[3], Numeric],
    vector2_end : np.ndarray[Shape[3], Numeric],
    *,
    t1 : float=0.5,
    t2 : float=None,
) -> RigidTransform:
    '''
    Compute a rigid transformation that forces "vector1" (defined by its end points and not necessarily emanating from the origin)
    onto the span of "vector2" (defined similarly), oriented parallel and with the point some fraction "t1" of the way along vector1
    to be exactly coincident with the point t2 parts along vector2
    '''
    # TODO: check compatibility of shapes within AND between vectors
    if t2 is None:
        t2 = t1
    
    vector1 = vector1_end - vector1_start
    overlaid_point_1 = vector1_start + t1*vector1
    
    vector2 = vector2_end - vector2_start
    overlaid_point_2 = vector2_start + t2*vector2
    
    return (
        RigidTransform.from_translation(overlaid_point_2)
        * RigidTransform.from_rotation(alignment_rotation(vector1, vector2))
        * RigidTransform.from_translation(-overlaid_point_1)
    )