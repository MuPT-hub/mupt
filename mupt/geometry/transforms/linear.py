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

def planar_reflector(normal_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    '''Computes a linear transformation which, when applied to an arbitrary vector,
    reflects it across the plane defined by the normal vector provided
    
    Returns an orthogonal Householder matrix which represents the transformation
    '''
    (dim,) = normal_vector.shape # extract dimension of vector space (and implicitly enforce 1D array)
    return np.eye(dim, dtype=normal_vector.dtype) - 2*parallel_projector(normal_vector) # compute Householder reflection

# DEVNOTE: deprecate this, or reimplement this correctly (i.e. preserve handedness)
def alignment_transform(initial_vector : np.ndarray[Shape[Dims], Numeric], final_vector : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims, Dims], Numeric]:
    # TODO: rename this to something more accurate, i.e. "Householder Reflector"
    '''Computes an orthogonal linear transformation which reflects initial_vector onto final_vector
    while preserving the relative orientations of the basis vectors to one another
    
    Returns an orthogonal Householder matrix which represents the transformation'''
    assert final_vector.shape == initial_vector.shape # ensure both vectors have 
    # DEVNOTE: worth verifying that magnitudes of vectors are the same? (or just be content with matching span)

    return planar_reflector(final_vector - initial_vector)

def axial_rotator(axis_vector : np.ndarray[Shape[Dims], Numeric], angle_rad : float=0.0) -> np.ndarray[Shape[Dims, Dims], Numeric]:
   # DEVNOTE: should type annotations be for general dimension? (i.e. not just 3 where the cross product is well-defined?)
   ''' 
   Computes a linear transformation which, when applied to an arbitrary vector,
   rotates that vector by "angle_rad" radians around the axis defined by "axis_vector"
   (in a right-handed coordinate systems), as calculated by Rodrigues' rotation formula.
   
   Returns an orthogonal matrix which represents the rotation transformation. 
   '''
   (dims,) = axis_vector.shape # implicitly enforce 1D rotation axis
   I = np.eye(dims, dtype=axis_vector.dtype)
   axis_cross = np.cross(I, axis_vector / np.linalg.norm(axis_vector)) # linear transform equivalent to taking the cross product with the (unit) axis vector
   
   return I + np.sin(angle_rad)*axis_cross + (1 - np.cos(angle_rad))*(axis_cross @ axis_cross)


