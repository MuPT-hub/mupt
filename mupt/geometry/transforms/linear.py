'''Application and construction of common linear transformations of points in 3D space'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from scipy.spatial.transform import Rotation

from ..arraytypes import Shape, N, Dims, Numeric


def parallel_projector(direction_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector parallel to the direction vector provided
    
    Returns matrix which represents the parallel projector transformation'''
    (dim,) = direction_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    return np.outer(direction_vector, direction_vector) / np.inner(direction_vector, direction_vector)

def orthogonal_projector(direction_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''Computes a linear transformation which, when applied to an arbitrary vector,
    yields the component of that vector orthogonal to the direction vector provided
    
    Returns matrix which represents the orthogonal projector transformation'''
    (dim,) = direction_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    return np.eye(dim, dtype=direction_vector.dtype) - parallel_projector(direction_vector) # equivalent to substracting parallel part off of vector transform is applied to
perpendicular_projector = orthogonal_projector # alias for convenience

def axial_rotator(axis_vector : np.ndarray[Shape[Dims], Numeric], angle_rad : float=0.0) -> np.ndarray[Shape[Dims, Dims], Numeric]:
   # DEVNOTE: should type annotations be for general dimension? (i.e. not just 3 where the cross product is well-defined?)
   ''' 
   Computes a linear transformation which, when applied to an arbitrary vector,
   rotates that vector by "angle_rad" radians around the axis defined by "axis_vector"
   (in a right-handed coordinate systems)
   
   Returns an orthogonal matrix which represents the rotation transformation. 
   Calculation is based on the Rodrigues' rotation formula.
   '''
   (dims,) = axis_vector.shape # implicitly enforce 1D rotation axis
   I = np.eye(dims, dtype=axis_vector.dtype)
   axis_cross = np.cross(I, axis_vector / np.linalg.norm(axis_vector)) # linear transform equivalent to taking the cross product with the (unit) axis vector
   
   return I + np.sin(angle_rad)*axis_cross + (1 - np.cos(angle_rad))*(axis_cross @ axis_cross)

def alignment_transform(initial_vector : np.ndarray[Shape[Dims], Numeric], final_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    # TODO: rename this to something more accurate, i.e. "Householder Reflector"
    '''Computes an orthogonal linear transformation which takes a given initial vector to a final vector
    while preserving the relative orientations of the basis vectors to one another
    
    Returns an orthogonal Householder matrix which represents the transformation'''
    (dim,) = initial_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    assert final_vector.shape == initial_vector.shape # ensure both vectors have 
    diff = final_vector - initial_vector # could arbitrarily reverse order, will only differ in sign of reflection plane normal
    
    return np.eye(dim, dtype=initial_vector.dtype) - 2*parallel_projector(diff) # compute Householder reflection
