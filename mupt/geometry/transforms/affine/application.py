'''Utilities for applying affine transformations to other objects (not necessarily just points!)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Literal, Mapping, Optional, Sequence, Union
from typing import Protocol, runtime_checkable

import numpy as np

from .homogeneous import from_homogeneous_coords, to_homogeneous_coords
from ...arraytypes import Shape, Numeric, N, Dims, DimsPlus


@runtime_checkable
class AffineTransformable(Protocol):
    '''Interface for objects that can undergo an affine transformation'''
    def affine_transformation(self, transformation: np.ndarray[Shape[N, N], float]) -> Any: 
        # DEVNOTE: regarding typehints, returned type may be different to type of self, and is not necessarily transformable either
        ...

def apply_affine_transformation_recursive(
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
            apply_affine_transformation_recursive(value, affine_matrix)
                for value in obj
        ) 
    elif isinstance(obj, Mapping):
        return {
            key : apply_affine_transformation_recursive(value, affine_matrix)
                for (key, value) in obj.items()
        }
        
    return obj

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
