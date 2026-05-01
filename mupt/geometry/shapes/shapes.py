'''Interfaces for shaped objects and other types which handle them'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional, Protocol, runtime_checkable
from abc import abstractmethod

from ..arraytypes import (
    NumberLike,
    Vector3,
    ArrayNxN,
    ArrayNx3,
    TriangulationIndices,
)
from ..transforms.rigid.application import RigidlyTransformable


@runtime_checkable # TODO: make class which composes transformability and boundedness (not necessarily the same a priori)
class BoundedShape(Protocol):
    '''Interface for bounded rigid bodies which can undergo coordinate transforms'''
    # measures of extent
    @property
    @abstractmethod
    def centroid(self) -> Vector3:
        '''Coordinate of the geometric center of the body'''
        ...
    # COM = CoM = center_of_mass = centroid # aliases for convenience
    
    @property
    @abstractmethod
    def volume(self) -> NumberLike:
        '''Cumulative measure within the boundary of the body'''
        ...
        
    @abstractmethod
    def contains(self, points : Vector3 | ArrayNxN) -> bool: 
        '''Whether a given coordinate lies within the boundary of the body'''
        ... 

    @abstractmethod
    def surface_mesh(self, *args, **kwargs) -> tuple[ArrayNx3, TriangulationIndices]:
        '''
        Generate a triangulated mesh of the surface of the shape which can be easily digested and plotted by mpl.plot_trisurf
        
        Returns
        -------
        mesh_points : Array[[N, 3], Numeric]
            An Nx3 array of the XYZ positions of each point in the mesh
        tri_vertices : Array[[T, 3], int]
            A Tx3 array describing the T triples in the mesh, with each row being the triples of array indices of that triangle
            For example, a row with [1,3,6] represents the triangle traversed counterclockwise from vertices 1 -> 3 -> 6 -> 1
        '''
        ...
    
    # @abstractmethod
    # def support(self, direction : np.ndarray[Shape[3], Numeric]) -> np.ndarray[Shape[3], Numeric]:
    #     '''Determines the furthest point on the surface of the body in a given direction'''
    #     ...

class BoundedTransformableShape(BoundedShape, RigidlyTransformable):
    '''Interface for bounded rigid bodies which can undergo coordinate transforms'''
    ...
        
class Shaped(Protocol):
    '''Interface for objects which have an associated bounded, tranformable shape'''
    _shape : Optional[BoundedTransformableShape]
    
    @property
    def has_shape(self) -> bool:
        '''Whether this Primitive has an associated external shape'''
        return self._shape is not None
    
    @property
    def shape(self) -> Optional[BoundedTransformableShape]: # TODO: make ShapedPrimitive subtype to avoid all these None checks?
        '''The external shape of this Primitive'''
        return self._shape
    
    @shape.setter
    def shape(self, new_shape : Optional[BoundedTransformableShape]) -> None:
        '''Set the external shape of this Primitive with another BoundedShape'''
        # Case 1) no shape
        if new_shape is None:
            self._shape = None
            return
        
        # Case 2) valid shape, which may need to have transformation history transferred over
        if not isinstance(new_shape, BoundedTransformableShape):
            raise TypeError(f'Primitive shape must be BoundedShape instance, not object of type {type(new_shape.__name__)}')

        new_shape_clone = new_shape.copy() # NOTE: make copy to avoid mutating original (per Principle of Least Astonishment)
        if self._shape is not None:
            new_shape_clone.cumulative_transformation = self._shape.cumulative_transformation # transfer translation history BEFORE overwriting
        
        self._shape = new_shape_clone
        