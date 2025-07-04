'''For encoding rigid bodies in space'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generic, Optional, Sequence

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from .arraytypes import Shape, Numeric, M, N, P, Dims
from .coordinates.basis import is_columnspace_mutually_orthogonal
from .transforms.affine import (
    affine_matrix_from_linear_and_center,
    apply_affine_transformation_to_points,
)


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
    
    def sample(self, radius : Numeric=1.0, num_points : int=1) -> np.ndarray[Shape[N, 3], Numeric]:
        '''Sample a random point from the plane within a given distance from the origin in the XY-plane (default 1 unit)'''
        x = np.random.uniform(-radius, radius, size=num_points)
        y = np.random.uniform(-radius, radius, size=num_points)
        z = - (self.a*x + self.b*y + self.d)/(self.c) # z in constrained by first 2 choices
        
        return np.column_stack([x, y, z])


class BoundedShape(ABC, Generic[Numeric]): # template for numeric type (some iterations of float in most cases)
    '''Interface for bounded rigid bodies which can undergo coordinate transforms'''
    @property
    @abstractmethod
    def centroid(self) -> np.ndarray[Shape[Dims], Numeric]:
        '''Coordinate of the geometric center of the body'''
        ...
    # COM = CoM = center_of_mass = centroid # aliases for convenience
    
    @property
    @abstractmethod
    def volume(self) -> Numeric:
        '''Cumulative measure within the boundary of the body'''
        ...
        
    @abstractmethod
    def contains(self, point : np.ndarray[Shape[Dims], Numeric]) -> bool: # TODO: enforce generalization to vectors of coordinates, rather than individual points
        '''Whether a given coordinate lies within the boundary of the body'''
        ... 
        
    @abstractmethod
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[N, N], Numeric]) -> 'BoundedShape':
        '''Implemenation of how the body should actually apply an affine transformation matrix'''
        ...

    def affine_transformation(self, affine_matrix : np.ndarray[Shape[N, N], Numeric]) -> 'BoundedShape':
        '''
        Apply an affine transformation to the body, as encoded by a transformation matrix
        Matrix should be square and have dimension exactly one greater that that of the body
        '''
        assert(affine_matrix.shape == (self.dimension + 1, self.dimension + 1)) # enforce squareness and dimensionality of transform
        return self._apply_affine_transformation(affine_matrix)
     
    # @abstractmethod
    # def support(self, direction : np.ndarray[Shape[Dims], Numeric]) -> np.ndarray[Shape[Dims], Numeric]:
    #     '''Determines the furthest point on the surface of the body in a given direction'''
    #     ...
        

# Concrete BoundedShape implementations
class PointCloud(BoundedShape[Numeric]):
    '''A cluster of points in 3D space'''
    def __init__(self, positions : np.ndarray[Shape[3], Numeric]=None) -> None:
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
    def centroid(self) -> np.ndarray[Shape[3], Numeric]:
        return self.positions.mean(axis=0)
    
    @property
    def volume(self) -> Numeric:
        return self.convex_hull.volume
    
    def contains(self, point : np.ndarray[Shape[3], Numeric]) -> bool:
        return (self.triangulation.find_simplex(point) != -1).astype(object) # need to cast from numpy bool to Python bool
    
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[4, 4], Numeric]) -> 'PointCloud':
        return PointCloud(
            positions=apply_affine_transformation_to_points(self.positions, affine_matrix)
        )
    
# @dataclass
class Ellipsoid(BoundedShape[Numeric]):
    '''
    A generalized spherical body, with potentially asymmetric orthogonal principal axes and arbitrary centroid
    
    Represented by a (not necessarily isotropic) scaling of the basis vectors and a rigid transformation,
    which, together, map the points on a unit sphere at the origin to the surface of the ellipsoid
    '''
    # initialization
    def __init__(self, basis : np.ndarray[Shape[4, 4], Numeric]=None) -> None:
        if basis is None:
            basis = np.eye(4, dtype=float)
            
        assert self.is_valid_ellipsoid_matrix(basis)
        self.basis = basis
        
    def __repr__(self):
        return f'{self.__class__.__name__}(radii={self.radii}, center={self.centroid})'
        
    @cached_property
    def basis_inverse(self) -> np.ndarray[Shape[4, 4], Numeric]:
        '''Transformation which maps this ellipsoid to the unit sphere centered at the origin
        Cached to avoid matrix inverse recalculation'''
        return np.linalg.inv(self.basis) # precompute inverse for later use
    
    @staticmethod
    def is_valid_ellipsoid_matrix(basis : np.ndarray[Shape[4, 4], Numeric]) -> bool:
        '''Check that an affine matrix could represent an ellipsoid'''
        assert basis.shape == (4, 4)
        axes, center, projective_part, w = basis[:-1, :-1], basis[:-1, -1], basis[-1, :-1], basis[-1, -1] # TODO: find more elegant way to do this splitting
        
        return bool(
            is_columnspace_mutually_orthogonal(axes) # ensure principal axes are mutually orthogonal
            and np.allclose(projective_part, 0.0) # ensure axes have apply no projective transformation
            and np.isclose(w, 1.0), # ensure homogeneous scale of the center is 1 (i.e. unprojected)
        )
        
    @classmethod
    def from_axes_and_center(
        cls,
        axes : np.ndarray[Shape[3, 3], Numeric],
        center : Optional[np.ndarray[Shape[3], Numeric]]=None,
    ) -> 'Ellipsoid':
        '''Instantiate an ellipsoid from a matrix of its axes and
        an (optional) location for its center (by default, the origin)'''
        # NOTE: explicitly using Ellipsoid (rather than cls) to prevent circular call in child classes (i.e. Sphere)
        return Ellipsoid(basis=affine_matrix_from_linear_and_center(axes, center=center))
    
    @classmethod
    def from_axis_lengths_and_center(
        cls,
        radius_x : float=1.0,
        radius_y : float=1.0,
        radius_z : float=1.0, 
        center : Optional[np.ndarray[Shape[3], Numeric]]=None,
    ) -> 'Ellipsoid':
        '''Instantiate an ellipsoid its axis lengths andan (optional) location for its center (by default, the origin)'''
        # TODO: add some "smart" conversion of arrays/diagonal matrices as input
        return cls.from_axes_and_center(
            axes=np.diag([radius_x, radius_y, radius_z]),
            center=center
        )
        
    # interfaces
    @property
    def centroid(self) -> np.ndarray[Shape[3], Numeric]:
        return self.basis[:-1, -1] # return translation vector of center
    
    @property
    def volume(self) -> Numeric:
        return 4/3 * np.pi * np.linalg.det(self.basis)
    
    def contains(self, point : np.ndarray[Shape[3], Numeric]) -> bool:   # TODO: decide whether containment should be boundary-inclusive
        return (np.linalg.norm(
            apply_affine_transformation_to_points(
                positions=point,
                transform=self.basis_inverse, # inverse transform maps all points inside the ellipsoid to points within the unit ball
            ),
            axis=-1,
        ) < 1).astype(object) # need to cast from numpy bool to Python bool
    
    def _apply_affine_transformation(self, affine_matrix : np.ndarray[Shape[4, 4], Numeric]) -> 'Ellipsoid':
        return Ellipsoid(affine_matrix @ self.basis)
    
    # visualization
    @property
    def radii(self) -> np.ndarray[Shape[3], Numeric]:
        '''The lengths of the principal axes of the ellipsoid'''
        return np.linalg.norm(self.basis[:-1, :-1], axis=0)
        # TODO: rewrite as transpose product, exploiting OD decomposition of valid ellipsoid matrix
    axes_lengths = radii # alias for convenience
    
    def surface_mesh(self, n_theta : int=100, n_phi : int=100) -> np.ndarray[Shape[M, P, 3], Numeric]:
        '''
        Generate a mesh of points on the surface of the ellipsoid
        
        Parameters
        ----------
        n_theta : int, default 100
            Number of points in the azimuthal angle direction
            Equivalent to longitudinal resolution
            
            Theta is taken to be the angle CC from the +x axis in the xy-plane,
            following the mathematics (not physics!) convention
        n_phi : int, default 100
            Number of points in the polar angle direction
            Equivalent to latitudinal resolution
            
            Phi is taken to be the angle "downwards" from the +z axis
            following the mathematics (not physics!) convention
            
        Returns
        -------
        ellipsoid_mesh : Array[[M, P, 3], float]
            A mesh of points on the surface of the ellipsoid
            M is the number of points in the azimuthal direction
            P is the number of points in the polar direction
        '''
        r : float = 1.0 # NOTE: this is NOT a parameter, but is left here to make clear tht we start with a UNIT sphere
        theta, phi = np.mgrid[
            0.0:2*np.pi:n_theta*1j,
            0.0:np.pi:n_phi*1j,
        ] # (magnitude of) complex step size is interpreted by numpy as a number of points

        positions = np.zeros((n_theta, n_phi, 3), dtype=Numeric)
        positions[..., 0] = r * np.sin(phi) * np.cos(theta)
        positions[..., 1] = r * np.sin(phi) * np.sin(theta)
        positions[..., 2] = r * np.cos(phi)
        
        return apply_affine_transformation_to_points(positions, self.basis)
    
class Sphere(Ellipsoid[Numeric]):
    '''A spherical body with arbitrary radius and center'''
    def __init__(self, radius : float=1.0, center : np.ndarray[Shape[3], Numeric]=None) -> 'Sphere':
        super().__init__(affine_matrix_from_linear_and_center(
            matrix=radius * np.eye(3, dtype=float),
            center=center,
        ))
        self.radius = radius
        self.center = center
    
    def __repr__(self):
        return f'Sphere(r={self.radius})'
    
    # TODO: address creation from axes (https://en.wikipedia.org/wiki/Circle%E2%80%93ellipse_problem)
    
    # NOTE: affine transformations will produce Ellipsoid instances, as one would expect
