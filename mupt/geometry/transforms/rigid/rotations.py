'''Utilities for handling proper rotations (i.e. elements of the special orthogonal group SO(3))'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from scipy.spatial.transform import Rotation

from ..linear import reflector, orthogonalizer
from ...arraytypes import Shape, Numeric
from ...measure import normalized

from ...coordinates.basis import is_orthogonal


def rotator(
    rotation_axis : np.ndarray[Shape[3], Numeric],
    angle_rad : float=0.0,
) -> Rotation:
    ''' 
    NOTE: ADVISE USING Rotation.from_rotvec(normalized(rotation_axis) * angle_rad) INSTEAD
    
    Computes a linear transformation which, when applied to an arbitrary vector,
    rotates that vector by "angle_rad" radians around the axis defined by "rotation_axis"
    (in a right-handed coordinate systems), as calculated by Rodrigues' rotation formula.
    
    Returns an orthogonal matrix which represents the rotation transformation. 
    '''
    (dims,) = rotation_axis.shape # implicitly enforce 1D shape for vector
    I = np.eye(dims, dtype=rotation_axis.dtype)
    K = orthogonalizer(rotation_axis)
    
    return Rotation.from_matrix(
        I + np.sin(angle_rad)*K + (1 - np.cos(angle_rad))*(K @ K)
    )
rodrigues = rotator

def alignment_rotation(
    moved_vector : np.ndarray[Shape[3], Numeric],
    onto_vector : np.ndarray[Shape[3], Numeric],
) -> Rotation:
    '''
    Compute a rotation which takes moved_vector parallel to the span of onto_vector
    Implemented as a composition of 2 Householder reflections to avoid any explicit angle calculations
    '''
    ## double reflection ensures handedness of basis is preserved; have found that reflection about bisector (mean)
    ## axis is more numerically stable than about the difference axis, especially for nearly-identical vectors
    
    ## bisector <=> vector which bisects the angle between the pair of vectors;
    ## proportional to the mean of any pair of equal length vectors on the two vectors' respective spans,
    ## e.g. the sum of normal vectors on the two spans will do the trick
    if np.allclose(moved_vector, onto_vector): # special case to avoid numerical errors
        return Rotation.identity() # no rotation needed
    
    bisector = normalized(onto_vector) + normalized(moved_vector) 
    rotation_matrix = reflector(onto_vector) @ reflector(bisector) # onto_vector doesn't need to be normalized
    assert is_orthogonal(rotation_matrix), 'Calculated alignment is not a proper rotation'

    return Rotation.from_matrix(rotation_matrix)