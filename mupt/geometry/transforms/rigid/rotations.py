'''Utilities for handling proper rotations (i.e. elements of the special orthogonal group SO(3))'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..linear import reflector, orthogonalizer
from ...arraytypes import Shape, Numeric
from ...measure import normalized
from ...coordinates.directions import random_orthogonal_vector


def rotator(rotation_axis : np.ndarray[Shape[3], Numeric], angle_rad : float=0.0) -> Rotation:
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
        initial_vector : np.ndarray[Shape[3], Numeric],
        final_vector   : np.ndarray[Shape[3], Numeric],
        orthogonal_vector : Optional[np.ndarray[Shape[3], Numeric]]=None,
    ) -> Rotation:
    '''
    Compute a rotation which aligns initial_vector to final_vector, preserving orientation
    Can optionally provide a vector orthogonal to initial_vector to define the local coordinate system;
    If none is provided, an orthogonal vector will be selected randomly instead
    
    Implemented as a composition of 2 Householder reflections to avoid
    any angle calculations with inverse trigonometric functions
    '''
    if orthogonal_vector is None:
        orthogonal_vector = random_orthogonal_vector(initial_vector)
        
    if not np.isclose(np.dot(initial_vector, orthogonal_vector), 0.0):
        raise ValueError('Orthogonal vector must be orthogonal to the initial vector')

    ## pre-reflect to invert handedness without moving the initial vector...
    ## ...then align by reflecting along the bisecting plane, restoring handedness
    preflip_reflection = reflector(np.cross(initial_vector, orthogonal_vector))
    alignment_reflection = reflector(normalized(initial_vector) - normalized(final_vector))
    rotation_matrix = alignment_reflection @ preflip_reflection
    assert np.isclose(np.linalg.det(rotation_matrix), 1.0), 'Proper rotation must have determinant 1.0'
    
    return Rotation.from_matrix(rotation_matrix)