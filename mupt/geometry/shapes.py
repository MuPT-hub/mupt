'''For encoding rigid bodies in space'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generic, Sequence, TypeVar

T = TypeVar('T')

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from .homogeneous import apply_affine_transform_to_points
from ..mutils.decorators.classmod import register_abstract_class_attrs
from .arraytypes import Shape, Numeric, DType, M, N, Dims, Vector3, ArrayNxN

type Point3D = np.ndarray[Shape[3], Numeric]


@dataclass
class Plane(Generic[Numeric]):
    '''
    Represents a plane in 3-space
    Represents the locus of points (x, y, z) satisfying a*x + b*y + c*z + d = 0
    '''
    a : Numeric
    b : Numeric
    c : Numeric
    d : Numeric = 0
    
    @classmethod
    def from_normal_and_point(cls, normal : Point3D, point : Point3D) -> 'Plane':
        '''Initialize from a normal vector and an arbitrary point know to lie in the plane'''
        assert isinstance(point, np.ndarray) and point.size == 3
        a, b, c = normal
        
        assert isinstance(point, np.ndarray) and point.size == 3
        d = np.dot(normal, point)
        
        return cls(a, b, c, -d)
    
    @property
    def normal(self) -> Point3D:
        return np.array([self.a, self.b, self.c])
    
    def contains(self, *point : Sequence[Numeric]) -> bool:
        '''Test whether a point lies on the plane defined'''
        if len(point) == 1 and isinstance(point[0], (Sequence, np.ndarray)):
            point = point[0] # correct missing star-args for Sequence-like
        
        assert len(point) == 3
        x, y, z = point
        
        return np.isclose(self.a*x + self.b*y + self.c*z + self.d, 0.0).astype(object) # convert from Numpy to Python bool
    
    def sample(self, radius : Numeric=1.0, n : int=1) -> Point3D:
        '''Sample a random point from the plane within a given distance from the origin in the XY-plane (default 1 unit)'''
        x = np.random.uniform(-radius, radius, size=n)
        y = np.random.uniform(-radius, radius, size=n)
        z = - (self.a*x + self.b*y + self.d)/(self.c) # z in constrained by first 2 choices
        
        return np.column_stack([x, y, z])


@register_abstract_class_attrs('dimension') # requires that subclasses implement a dimensionality at the class level
class BoundedShape(ABC, Generic[T]): # template for numeric type (some iterations of float in most cases)
    '''Interface for bounded rigid bodies which can undergo coordinate transforms'''
    @property
    @abstractmethod
    def centroid(self) -> np.ndarray[Shape[Dims], T]:
        '''Coordinate of the geometric center of the body'''
        ...
    # COM = CoM = center_of_mass = centroid # aliases for convenience
    
    @property
    @abstractmethod
    def volume(self) -> T:
        '''Cumulative measure within the boundary of the body'''
        ...
        
    @abstractmethod
    def contains(self, point : np.ndarray[Shape[Dims], T]) -> bool:
        '''Whether a given coordinate lies within the boundary of the body'''
        ... 
        
    @abstractmethod
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[N, N], T]) -> 'BoundedShape':
        '''Implemenation of how the body should actually apply an affine transformation matrix'''
        ...

    def affine_transformation(self, affine_matrix : np.ndarray[Shape[N, N], T]) -> 'BoundedShape':
        '''
        Apply an affine transformation to the body, as encoded by a transformation matrix
        Matrix should be square and have dimension exactly one greater that that of the body
        '''
        assert(affine_matrix.shape == (self.dimension + 1, self.dimension + 1)) # enforce squareness and dimensionality of transform
        return self._apply_affine_transformation(affine_matrix)
     
    # @abstractmethod
    # def support(self, direction : np.ndarray[Shape[Dims], T]) -> np.ndarray[Shape[Dims], T]:
    #     '''Determines the furthest point on the surface of the body in a given direction'''
    #     ...

# Concrete BoundedShape implementations
@dataclass
class Sphere(BoundedShape[float], dimension=3):
    '''A spherical body with arbitrary radius and center'''
    radius : float
    center : np.ndarray[Shape[3], float]
    
    @property
    def centroid(self) -> np.ndarray[Shape[3], float]:
        return self.center
    
    @property
    def volume(self) -> float:
        return 4/3 * np.pi * self.radius**3
    
    def contains(self, point : np.ndarray[Shape[3], float]) -> bool:   # TODO: decide whether containment should be boundary-inclusive
        return bool(np.linalg.norm(self.center - point) < self.radius) # need to cast from numpy bool to Python bool
    
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[4, 4], T]) -> 'Sphere':
        return Sphere( # TODO: should return an ellipsoid if scaling in anisotropically
            radius=self.radius * np.linalg.det(affine_matrix)**(1/3), # scale radius appropriately
            center=apply_affine_transform_to_points(self.center, affine_matrix),
        ) # TODO: have non-isometric affine transforms correctly return an Ellipsoid, once implemented

# TODO: implement Ellipsoid subclass

@dataclass
class PointCloud(BoundedShape[float], dimension=3):
    '''A cluster of points in 3D space'''
    coordinates : np.ndarray[Shape[3], float]
    _convex_hull : ConvexHull = field(init=False, default=None)
    _triangulation : Delaunay = field(init=False, default=None)
    
    @property
    def convex_hull(self) -> ConvexHull:
        if self._convex_hull is None:
            self._convex_hull = ConvexHull(self.coordinates)
        return self._convex_hull

    @property
    def triangulation(self) -> Delaunay:
        if self._triangulation is None:
            self._triangulation = Delaunay(self.coordinates)
        return self._triangulation
    
    @property
    def centroid(self) -> np.ndarray[Shape[3], float]:
        return self.coordinates.mean(axis=0)
    
    @property
    def volume(self) -> float:
        return self.convex_hull.volume
    
    def contains(self, point : np.ndarray[Shape[3], float]) -> bool:
        return bool(self.triangulation.find_simplex(point) != -1) # need to cast from numpy bool to Python bool
    
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[4, 4], T]) -> 'PointCloud':
        return PointCloud(
            coordinates=apply_affine_transform_to_points(self.coordinates, affine_matrix)
        )
    
