'''For encoding rigid bodies in space'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generic, Optional, Sequence, TypeVar

T = TypeVar('T')

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from ..mutils.decorators.classmod import register_abstract_class_attrs
from .arraytypes import Shape, Numeric, M, N, Dims
from .coordinates.homogeneous import apply_affine_transform_to_points
from .coordinates.basis import is_orthogonal


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
    def from_normal_and_point(cls, normal : np.ndarray[Shape[3], Numeric], point : np.ndarray[Shape[3], Numeric]) -> 'Plane':
        '''Initialize from a normal vector and an arbitrary point know to lie in the plane'''
        assert isinstance(point, np.ndarray) and point.size == 3
        a, b, c = normal
        
        assert isinstance(point, np.ndarray) and point.size == 3
        d = np.dot(normal, point)
        
        return cls(a, b, c, -d)
    
    @property
    def normal(self) -> np.ndarray[Shape[3], Numeric]:
        return np.array([self.a, self.b, self.c])
    
    def contains(self, *point : Sequence[Numeric]) -> bool:
        '''Test whether a point lies on the plane defined'''
        if len(point) == 1 and isinstance(point[0], (Sequence, np.ndarray)):
            point = point[0] # correct missing star-args for Sequence-like
        
        assert len(point) == 3
        x, y, z = point
        
        return np.isclose(self.a*x + self.b*y + self.c*z + self.d, 0.0).astype(object) # convert from Numpy to Python bool
    
    def sample(self, radius : Numeric=1.0, n : int=1) -> np.ndarray[Shape[3], Numeric]:
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
    def contains(self, point : np.ndarray[Shape[Dims], T]) -> bool: # TODO: enforce generalization to vectors of coordinates, rather than individual points
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
    
    # @classmethod
    # @abstractmethod
    # def from_coordinates(cls, positions : np.ndarray[Shape[N, Dims], T]) -> 'BoundedShape':
    #     '''Initialize a body from a list of positions'''
    #     ...
        

# Concrete BoundedShape implementations
class PointCloud(BoundedShape[float], dimension=3):
    '''A cluster of points in 3D space'''
    def __init__(self, positions : np.ndarray[Shape[3], float]=None) -> None:
        if positions is None:
            positions = np.empty((0, 0))
        self.positions = positions
        
    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.positions.shape})'

    @cached_property
    def convex_hull(self) -> ConvexHull:
        '''Convex hull of the points contained within'''
        return ConvexHull(self.positions)

    @cached_property
    def triangulation(self) -> Delaunay:
        '''Delauney triangulation into simplicial facets whose vertiecs are the positions within'''
        return Delaunay(self.positions)
    
    @property
    def centroid(self) -> np.ndarray[Shape[3], float]:
        return self.positions.mean(axis=0)
    
    @property
    def volume(self) -> float:
        return self.convex_hull.volume
    
    def contains(self, point : np.ndarray[Shape[3], float]) -> bool:
        return (self.triangulation.find_simplex(point) != -1).astype(object) # need to cast from numpy bool to Python bool
    
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[4, 4], T]) -> 'PointCloud':
        return PointCloud(
            positions=apply_affine_transform_to_points(self.positions, affine_matrix)
        )
    
@dataclass
class Ellipsoid(BoundedShape[float], dimension=3):
    '''A generalized spherical body, with potentially asymmetric orthogonal principal axes and arbitrary centroid
    Represented by an affine transformation of a unit sphere centered at the origin'''
    # affine matrix with principal basis as linear part and location of center as translational part
    def __init__(self, basis : np.ndarray[Shape[4, 4], float]=None) -> None:
        if basis is None:
            basis = np.eye(4, dtype=float)
            
        assert self.is_valid_ellipsoid_matrix(basis)
        self.basis = basis
        
    @cached_property
    def basis_inverse(self) -> np.ndarray[Shape[4, 4], float]:
        '''Transformation which maps this ellipsoid to the unit sphere centered at the origin
        Cached to avoid matrix inverse recalculation'''
        return np.linalg.inv(self.basis) # precompute inverse for later use
    
    @staticmethod
    def is_valid_ellipsoid_matrix(basis : np.ndarray[Shape[4, 4], float]) -> bool:
        '''Check that an affine matrix could represent an ellipsoid'''
        assert basis.shape == (4, 4)
        axes, center, projective_part, w = basis[:-1, :-1], basis[:-1, -1], basis[-1, :-1], basis[-1, -1] # TODO: find more leegant way to do this splitting
        
        return bool(
            is_orthogonal(axes) # ensure principal axes are mutually orthogonal
            and np.allclose(projective_part, 0.0) # ensure axes have apply no projective transformation
            and np.isclose(w, 1.0), # ensure homogeneous scale of the center is 1 (i.e. unprojected)
        )
        
    @classmethod
    def from_axes_and_center(
        cls,
        axes : np.ndarray[Shape[3, 3], float],
        center : Optional[np.ndarray[Shape[3], float]],
    ) -> 'Ellipsoid':
        '''Instantiate an ellipsoid from a matrix of its axes and
        an (optional) location for its center (by default, the origin)'''
        basis = np.zeros((4, 4), dtype=float)
        basis[:-1, :-1] = axes
        basis[:-1, -1]  = center
        basis[-1, -1]   = 1.0
        
        # return cls(basis=basis)
        return Ellipsoid(basis=basis) # NOTE: explicitly using Ellipsoid (rather than cls) to prevent circular call in child classes (i.e. Sphere)
        
    @property
    def centroid(self) -> np.ndarray[Shape[3], float]:
        return self.basis[:-1, -1] # return translation vector of center
    
    @property
    def volume(self) -> float:
        return 4/3 * np.pi * np.linalg.det(self.basis)
    
    def contains(self, point : np.ndarray[Shape[3], float]) -> bool:   # TODO: decide whether containment should be boundary-inclusive
        return (np.linalg.norm(
            apply_affine_transform_to_points(
                coords=point,
                transform=self.basis_inverse, # inverse transform maps all points inside the ellipsoid to points within the unit ball
            ),
            axis=-1,
        ) < 1).astype(object) # need to cast from numpy bool to Python bool
    
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[4, 4], T]) -> 'Ellipsoid':
        return Ellipsoid(affine_matrix @ self.basis)
    
class Sphere2(Ellipsoid):
    '''A spherical body with arbitrary radius and center'''
    def __new__(cls, radius : float=1.0, center : np.ndarray[Shape[3], float]=None, *args, **kwargs) -> 'Sphere2':
        if center is None:
            center = np.zeros(3, dtype=float)
            
        inst = super().from_axes_and_center(
            axes=radius * np.eye(3, dtype=float),
            center=center,
        )
        inst.radius = radius
        inst.center = center
        
        return inst
    
@dataclass # TODO: reimplement in terms of Ellipsoid
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
        
    # TODO: add mpl-compatible visualizer methods?
