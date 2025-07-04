'''Application and construction of common affine transformations'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Literal, Mapping, Optional, Sequence, Union
from typing import Protocol, runtime_checkable

import numpy as np

from ..arraytypes import Shape, Numeric, N, Dims, DimsPlus
type AffineMatrix4x4 = np.ndarray[Shape[Literal[4], Literal[4]], Numeric]

from ..coordinates.homogeneous import from_homogeneous_coords, to_homogeneous_coords


# INTERFACES FOR TRANFORMABLE OBJECTS
@runtime_checkable
class AffineTransformable(Protocol):
    '''Interface for objects that can undergo an affine transformation'''
    def affine_transformation(self, transformation: np.ndarray[Shape[N, N], float]) -> Any: 
        # DEVNOTE: regarding typehints, returned type may be different to type of self, and is not necessarily transformable either
        ...

def apply_affine_transform_recursive(
        obj : Union[object, Sequence[Any], Mapping[str, Any]],
        affine_matrix : np.ndarray[Shape[N, N], float],
    ) -> Union[object, Sequence[Any], dict[str, Any]]:
    '''Apply an affine transformation to an object, if it supports such a transformation,
    and, if the object is a Sequence or Mapping, attempt to transform its members recursively
    
    Parameters
    ----------
    obj : Any
        The object to be transformed, which may be a single object, a Sequence, or a Mapping
    affine_matrix : Array[[N, N], float]
        The affine transformation matrix to apply to the object
        
    Returns
    -------
    Any
        The transformed object, which (depending on the transformability of the input
        and its members and the return type of the transform method of members),
        may or many not be of the same type as the initial object
    '''
    # top-level application check
    if isinstance(obj, AffineTransformable):
        obj = obj.affine_transformation(affine_matrix)
        
    # recursive iteration, as necessary
    if isinstance(obj, Sequence):  # DEVNOTE: specifically opted for Sequence over Iterable here to avoid double-covering Mappings and unpacking generators
        return type(obj)( # DEVNOTE: most common Sequence types (e.g. tuple, str, list) support init from comprehension; may revisit if this is not always the case
            apply_affine_transform_recursive(value, affine_matrix)
                for value in obj
        ) 
    elif isinstance(obj, Mapping):
        return {
            key : apply_affine_transform_recursive(value, affine_matrix)
                for (key, value) in obj.items()
        }
        
    return obj
    

# APPLICATION OF AFFINE TRANSFORMATIONS
def apply_affine_transformation_to_points(
        positions : np.ndarray[Shape[Any, Dims], Numeric],
        transform : np.ndarray[Shape[DimsPlus, DimsPlus], Numeric],
    ) -> np.ndarray[Shape[Any, Dims], Numeric]:
    '''
    Take a vector of coordinates in D dimensions, apply a [D + 1] dimensional affine
    transformation, then project back down to D dimensions and return the output
    
    Parameters
    ----------
    positions : Array[[..., D], Numeric]
        An array of points in D-dimensional space
        Can be nested to any level of depth, as long as the last dimension is D
    transform : Array[[D + 1, D + 1], Numeric]
        A [D + 1] dimensional affine transformation matrix      
        
    Returns
    -------
    Array[[..., D], Numeric]
        An array of the transformed points in D-dimensional space
        Has the same shape as the input
    '''
    # TODO: check that matrix is compatible shape
    return from_homogeneous_coords(to_homogeneous_coords(positions) @ transform.T)

# CONSTRUCTION OF AFFINE MATRICES
def affine_matrix_from_linear_and_center(
        matrix : np.ndarray[Shape[Dims, Dims], Numeric],
        center : Optional[np.ndarray[Shape[Dims], Numeric]]=None,
        dtype : Optional[type]=None,
    ) -> np.ndarray[Shape[DimsPlus, DimsPlus], Numeric]:
        '''
        Instantiate an affine transformation matrix from a linear transformation and a new origin location
        
        Parameters
        ----------
        matrix : Array[[D, D], Numeric]
            A D-dimensional linear transformation matrix
        center : Array[[D,], Numeric]
            A D-dimensional vector representing the new origin location
        dtype : type
            The data type of the output matrix
            If None, will be the same as the input matrix
            
        Returns
        -------
        Array[[D + 1, D + 1], Numeric]
            The corresponding [D + 1] dimensional affine transformation matrix
        '''
        (n_rows, n_cols) = matrix.shape # implicitly enforces 2-dimensionality
        assert n_rows == n_cols # check squareness
        dimension = n_cols
        
        if dtype is None:
            dtype = matrix.dtype
        
        if center is None:
            center = np.zeros(dimension, dtype=dtype)
        
        affine_matrix = np.zeros((dimension + 1, dimension + 1), dtype=dtype)
        affine_matrix[:-1, :-1] = matrix
        affine_matrix[:-1, -1]  = center
        affine_matrix[-1, -1]   = 1
        
        return affine_matrix

def translation(x : float=0.0, y : float=0.0, z : float=0.0, dtype : Union[str, type]='float64') -> AffineMatrix4x4:
    '''
    Generates an affine matrix which translated the origin (and all points in space along with it) to the point (x, y, z)
    
    Parameters
    ----------
    x : float, default 1.0
        The x-coordinate of the translation
    y : float, default 1.0
        The y-coordinate of the translation
    z : float, default 1.0
        The z-coordinate of the translation
    dtype : type, default 'float64'
        The data type of the output matrix
        
    Returns
    -------
    translation_matrix : Array[[4, 4], float]
        The affine transformation matrix representing the translation
        With no arguments, returns the Identity matrix
    '''
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype=dtype)

def scaling(sx : float=1.0, sy : float=1.0, sz : float=1.0, dtype : Union[str, type]='float64') -> AffineMatrix4x4:
    '''
    Generates an affine matrix which scales the basis by factors 
    of (sx, sy, sz) along the x, y, and z axes, respectively
    
    Parameters
    ----------
    sx : float, default 1.0
        The scaling factor in the x-direction
    sy : float, default 1.0
        The scaling factor in the y-direction
    sz : float, default 1.0
        The scaling factor in the z-direction
    dtype : type, default 'float64'
        The data type of the output matrix
        
    Returns
    -------
    scaling_matrix : Array[[4, 4], float]
        The affine transformation matrix representing the scaling
        With no arguments, returns the Identity matrix
    '''
    return np.array([
        [sx,  0,  0, 0],
        [ 0, sy,  0, 0],
        [ 0,  0, sz, 0],
        [ 0,  0,  0, 1],
    ], dtype=dtype)

def rotation_x(angle_rad : float=0.0, dtype : Union[str, type]='float64') -> AffineMatrix4x4:
    '''
    Generates an affine matrix which rotates about the positive x-axis by "angle_rad" radians
    
    Parameters
    ----------
    angle_rad : float, default 0.0
        The angle of rotation about the x-axis, in radians
    dtype : type, default 'float64'
        The data type of the output matrix
        
    Returns
    -------
    rotation_matrix : Array[[4, 4], float]
        The affine transformation matrix representing the rotation
        With no arguments, returns the Identity matrix
    '''
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)
    
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ], dtype=dtype)

def rotation_y(angle_rad : float=0.0, dtype : Union[str, type]='float64') -> AffineMatrix4x4:
    '''
    Generates an affine matrix which rotates about the positive y-axis by "angle_rad" radians
    
    Parameters
    ----------
    angle_rad : float, default 0.0
        The angle of rotation about the y-axis, in radians
    dtype : type, default 'float64'
        The data type of the output matrix
        
    Returns
    -------
    rotation_matrix : Array[[4, 4], float]
        The affine transformation matrix representing the rotation
        With no arguments, returns the Identity matrix
    '''
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)
    
    return np.array([
        [c, 0, -s, 0],
        [0, 1,  0, 0],
        [s, 0,  c, 0],
        [0, 0,  0, 1],
    ], dtype=dtype)

def rotation_z(angle_rad : float=0.0, dtype : Union[str, type]='float64') -> AffineMatrix4x4:
    '''
    Generates an affine matrix which rotates about the positive z-axis by "angle_rad" radians
    
    Parameters
    ----------
    angle_rad : float, default 0.0
        The angle of rotation about the z-axis, in radians
    dtype : type, default 'float64'
        The data type of the output matrix
        
    Returns
    -------
    rotation_matrix : Array[[4, 4], float]
        The affine transformation matrix representing the rotation
        With no arguments, returns the Identity matrix
    '''
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)
    
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=dtype)

def rotation_random(about_x : bool=True, about_y : bool=True, about_z : bool=True, dtype : Union[str, type]='float64') -> AffineMatrix4x4:
    '''
    Generates an affine matrix which rotates by a random amount [0, 2pi)
    about any subset of the x, y, and z axes (including all 3 or neither)
    
    Parameters
    ----------
    about_x : bool, default True
        Whether to include a random rotation about the x-axis
    about_y : bool, default True
        Whether to include a random rotation about the y-axis
    about_z : bool, default True
        Whether to include a random rotation about the z-axis
    dtype : type, default 'float64'
        The data type of the output matrix
        If None, will be the same as the input matrix
    
    Returns
    -------
    rotation_matrix : Array[[4, 4], float]
        The affine transformation matrix representing the rotation
    '''
    rot_dir = { # concise way to encapsulate what rotations can be performed and whther to perform them
        rotation_x : about_x,
        rotation_y : about_y,
        rotation_z : about_z,
    }

    matrix = np.eye(4, dtype=float)
    for (rot_fn, should_rotate) in rot_dir.items():
        if should_rotate:
            matrix = rot_fn(2 * np.pi * np.random.rand(), dtype=dtype) @ matrix # generate random angle and multiply rotation into overall transform matrix (in order)

    return matrix
