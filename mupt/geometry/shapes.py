'''For encoding rigid bodies in space'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional, Sequence, Union

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation, RigidTransform

from .arraytypes import Shape, Numeric, M, N, P
from .coordinates.basis import (
    is_columnspace_mutually_orthogonal,
    is_orthogonal,
)
from .transforms.rigid.rotations import alignment_rotation
from .transforms.rigid.application import RigidlyTransformable


@dataclass
class Plane:
    '''
    Represents a plane in 3-space
    Represents the locus of points (x, y, z) satisfying a*x + b*y + c*z + d = 0
    '''
    a : Numeric
    b : Numeric
    c : Numeric
    d : Numeric = 0.0
    
    @classmethod
    def from_normal_and_point(cls,
        normal : np.ndarray[Shape[3], Numeric],
        point  : np.ndarray[Shape[3], Numeric],
    ) -> 'Plane':
        '''Initialize from a normal vector and an arbitrary point know to lie in the plane'''
        assert isinstance(point, np.ndarray) and point.size == 3
        a, b, c = normal
        
        assert isinstance(point, np.ndarray) and point.size == 3
        d = np.dot(normal, point)
        
        return cls(a, b, c, -d)
    
    @property
    def normal(self) -> np.ndarray[Shape[3], Numeric]:
        return np.array([self.a, self.b, self.c])
    
    def contains(self, *point : Sequence[Numeric]) -> bool: # TODO: support Nx3 arrays of points
        '''Test whether a point lies on the plane defined'''
        if len(point) == 1 and isinstance(point[0], (Sequence, np.ndarray)):
            point = point[0] # correct missing star-args for Sequence-like
        point = np.array(point, dtype=float)
        
        *_, ndim = point.shape
        assert ndim == 3
        
        return np.isclose(np.dot(point, self.normal) + self.d, 0.0).astype(object) # convert from Numpy to Python bool
    
    def sample(self, num_points : N=1, r_x : float=1.0, r_y : float=1.0) -> np.ndarray[Shape[N, 3], Numeric]:
        '''Sample a random point from the plane within a given distance from the origin in the XY-plane (default 1 unit)'''
        x = np.random.uniform(-r_x, r_x, size=num_points)
        y = np.random.uniform(-r_y, r_y, size=num_points)
        z = - (self.a*x + self.b*y + self.d)/(self.c) # z in constrained by first 2 choices
        
        return np.column_stack([x, y, z])
    
    def surface_mesh(self,
            center : Optional[np.ndarray[Shape[3], float]]=None,
            r_x : float=1.0,
            r_y : float=1.0,
            n_x : int=3,
            n_y : int=3,
        ) -> np.ndarray[Shape[M, P, 3], Numeric]:
        '''
        Generate a mesh of points on the surface of the plane
        
        Parameters
        ----------
        r_x : float, default 1.0
            Radius of the plane in the x direction
        r_y : float, default 1.0
            Radius of the plane in the y direction
        n_x : int, default 100
            Number of points in the x direction
        n_y : int, default 100
            Number of points in the y direction
            
        Returns
        -------
        plane_mesh : Array[[M, P, 3], float]
            A mesh of points on the surface of the plane
            
            M is the number of points in the x direction
            P is the number of points in the y direction
        '''
        if center is None:
            center = self.sample(num_points=1)
        assert self.contains(center), 'Center point must lie on the plane'
        
        # generate mesh of point for flat plane facing in z-direction
        x_flat, y_flat = np.mgrid[
            -r_x:r_x:n_x*1j,
            -r_y:r_y:n_y*1j,
        ]
        z_flat = np.zeros((n_x, n_y), dtype=float)
        points = np.dstack([x_flat, y_flat, z_flat])
        
        # rotate and translate mesh points into place based on plane parameters
        Z_UNIT = np.array([0, 0, 1], dtype=float) # unit vector in the z-direction
        plane_transform = RigidTransform.from_components(
            translation=center,
            rotation=alignment_rotation(Z_UNIT, self.normal),
        )

        return plane_transform.apply(points.reshape(-1, 3)).reshape(n_x, n_y, 3)


class BoundedShape(ABC, RigidlyTransformable): # template for numeric type (some iterations of float in most cases)
    '''Interface for bounded rigid bodies which can undergo coordinate transforms'''
    # measures of extent
    @property
    @abstractmethod
    def centroid(self) -> np.ndarray[Shape[3], Numeric]:
        '''Coordinate of the geometric center of the body'''
        ...
    # COM = CoM = center_of_mass = centroid # aliases for convenience
    
    @property
    @abstractmethod
    def volume(self) -> Numeric:
        '''Cumulative measure within the boundary of the body'''
        ...
        
    @abstractmethod
    def contains(self, points : np.ndarray[Union[Shape[3], Shape[N, 3]], Numeric]) -> bool: 
        '''Whether a given coordinate lies within the boundary of the body'''
        ... 

    # @abstractmethod
    # def support(self, direction : np.ndarray[Shape[3], Numeric]) -> np.ndarray[Shape[3], Numeric]:
    #     '''Determines the furthest point on the surface of the body in a given direction'''
    #     ...
        

# Concrete BoundedShape implementations
class PointCloud(BoundedShape):
    '''A cluster of points in 3D space'''
    def __init__(self, positions : np.ndarray[Shape[N, 3], Numeric]=None) -> None:
        if positions is None:
            positions = np.empty((0, 3), dtype=float)
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
    
    # fulfilling BoundedShape contracts
    @property
    def centroid(self) -> np.ndarray[Shape[3], Numeric]:
        return self.positions.mean(axis=0)
    
    @property
    def volume(self) -> Numeric:
        return self.convex_hull.volume
    
    def contains(self, points : np.ndarray[Union[Shape[3], Shape[N, 3]]]) -> bool:
        return (self.triangulation.find_simplex(points) != -1).astype(object) # need to cast from numpy bool to Python bool

    # fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'PointCloud':
        return self.__class__(positions=np.array(self.positions))

    def _rigidly_transform(self, transform : RigidTransform) -> None:
        self.positions = transform.apply(self.positions)
    
class Ellipsoid(BoundedShape):
    '''
    A generalized spherical body, with potentially asymmetric orthogonal principal axes and arbitrary centroid
    
    Representable by a (not necessarily isotropic) scaling of the basis vectors and a rigid transformation,
    which, together, map the points on a unit sphere at the origin to the surface of the Ellipsoid
    '''
    def __init__(
        self,
        radii : np.ndarray[Shape[3], Numeric]=None,
        center : np.ndarray[Shape[3], Numeric]=None,
    ) -> None:
        # DEV: extract this vector shape checking into external utility, eventually
        if radii is None:
            radii = np.ones(3, dtype=float)
        radii_std = np.atleast_2d(radii).reshape(-1) # permits transposed and nested vector inputs
        assert radii_std.shape == (3,)
            
        if center is None:
            center = np.zeros(3, dtype=float)
        center_std = np.atleast_2d(center).reshape(-1) # permits transposed and nested vector inputs
        assert center_std.shape == (3,)

        self.radii = radii
        self.center = center
        self.cumulative_transformation *= RigidTransform.from_translation(center)

    @classmethod
    def from_components(
        cls,
        # axis lengths
        radius_x : Numeric=1.0,
        radius_y : Numeric=1.0,
        radius_z : Numeric=1.0,
        # center coordinate
        center_x : Numeric=0.0,
        center_y : Numeric=0.0,
        center_z : Numeric=0.0,
    ) -> 'Ellipsoid':
        '''Instantiate Ellipsoid from array-wise representations of its radii and center'''
        return cls(
            radii=np.array([radius_x, radius_y, radius_z], dtype=float),
            center=np.array([center_x, center_y, center_z], dtype=float),
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(radii={self.radii}, center={self.center})'

    # Matrix representations of the Ellipsoid
    @staticmethod
    def is_valid_ellipsoid_matrix(basis : np.ndarray[Shape[4, 4], Numeric]) -> bool:
        '''Check that an affine matrix could represent an Ellipsoid'''
        assert basis.shape == (4, 4)
        axes, center, projective_part, w = basis[:-1, :-1], basis[:-1, -1], basis[-1, :-1], basis[-1, -1] # TODO: find more elegant way to do this splitting
        
        return bool(
            is_columnspace_mutually_orthogonal(axes) # ensure principal axes are mutually orthogonal
            and np.allclose(projective_part, 0.0) # ensure axes have apply no projective transformation
            and np.isclose(w, 1.0), # ensure homogeneous scale of the center is 1 (i.e. unprojected)
        )
        
    def scaling_matrix(self, as_affine : bool=True) -> np.ndarray[Union[Shape[3, 3], Shape[4, 4]], Numeric]:
        '''The scaling matrix which defines the radii of the Ellipsoid'''
        if as_affine:
            return np.diag([*self.radii, 1.0])  # add a 1.0 for the homogeneous coordinate
        return np.diag(self.radii)
        
    def as_affine_matrix(self) -> np.ndarray[Shape[4, 4], Numeric]:
        '''
        An affine matrix which represents this Ellipsoid
        
        Has the effect of transforming the unit sphere at the origin, 
        (in homogeneous coordinates) to the surface of this Ellipsoid
        '''
        return self.cumulative_transformation.as_matrix() @ self.scaling_matrix(as_affine=True)
    
    @property
    def basis(self) -> np.ndarray[Shape[4, 4], Numeric]:
        '''The basis matrix of the Ellipsoid - alias for Ellipsoid.as_affine_matrix()'''
        return self.as_affine_matrix()
    
    @property
    def principal_axes(self) -> np.ndarray[Shape[3, 3], Numeric]:
        '''The principal axes of the ellipsoid, represented as a 3x3 matrix
        whose rows are the axis vectors emanating from the Ellipsoid's center'''
        return self.cumulative_transformation.apply( self.scaling_matrix(as_affine=False) )
    axes = principal_axes # alias

    def affine_inverse(self) -> np.ndarray[Shape[4, 4], Numeric]:
        '''
        Transformation which maps this Ellipsoid to the unit sphere centered at the origin
        Inverse of the Ellipsoid's affine basis matrix
        '''
        return np.linalg.inv(self.as_affine_matrix) # precompute inverse for later use
    
    @property
    def inv(self) -> np.ndarray[Shape[4, 4], Numeric]:
        '''
        The inverse of the Ellipsoid's affine basis matrix - alias for Ellipsoid.affine_inverse()
        Maps this Ellipsoid to the unit sphere centered at the origin
        '''
        return self.affine_inverse()

    def coincident_with(self, other : 'Ellipsoid') -> bool:
        return np.allclose(self.radii, other.radii) \
            and np.allclose(self.center, other.center) \
            and np.allclose(self.cumulative_transformation.as_matrix(), other.cumulative_transformation.as_matrix())
        
    # fulfilling BoundedShape contracts
    @property
    def centroid(self) -> np.ndarray[Shape[3], Numeric]:
        return self.center
    
    @property
    def volume(self) -> Numeric:
        # return 4/3 * np.pi * np.linalg.det(self.matrix)
        return 4/3 * np.pi * np.prod(self.radii) # DEVNOTE: determination of rotation is always 1, so we may as well skip it and the whole determinant calculation

    def contains(self, points : np.ndarray[Union[Shape[3], Shape[N, 3]]]) -> bool:   # TODO: decide whether containment should be boundary-inclusive
        return (np.linalg.norm( # NOTE: not applying self.inverse to points because the Ellipsoid basis matrix is not, in general, a rigid transformation
            (self.resetting_transformation.apply(points) / self.radii), # reduce containment check to comparison with auxiliary unit sphere
            axis=1,
        ) < 1).astype(object) # need to cast from numpy bool to Python bool

    # fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'Ellipsoid':
        return self.__class__(
            radii=np.array(self.radii),
            center=np.array(self.center),
        )

    def _rigidly_transform(self, transform : RigidTransform) -> None:
        self.center = transform.apply(self.center)
    
    # visualization   
    def surface_mesh(self, n_theta : int=100, n_phi : int=100) -> np.ndarray[Shape[M, P, 3], Numeric]:
        '''
        Generate a mesh of points on the surface of this Ellipsoid
        
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
            A mesh of points on the surface of the Ellipsoid
            M is the number of points in the azimuthal direction
            P is the number of points in the polar direction
        '''
        r : float = 1.0 # NOTE: this is NOT a parameter, but is left here to make clear tht we start with a UNIT sphere
        theta, phi = np.mgrid[
            0.0:2*np.pi:n_theta*1j,
            0.0:np.pi:n_phi*1j,
        ] # (magnitude of) complex step size is interpreted by numpy as a number of points

        a, b, c = self.radii 
        positions = np.zeros((n_theta, n_phi, 3), dtype=float)
        positions[..., 0] = a * r * np.sin(phi) * np.cos(theta)
        positions[..., 1] = b * r * np.sin(phi) * np.sin(theta)
        positions[..., 2] = c * r * np.cos(phi)
        
        return self.cumulative_transformation.apply(
            positions.reshape(-1, 3) # flatten into (n_theta*n_phi)x3 array to allow RigidTransform.apply() to digest it
        ).reshape(n_theta, n_phi, 3) # ...then repackage into mesh for convenient plotting
    
    
class Sphere(Ellipsoid): 
    # TODO: reimplement as separate from Ellipsoid to address Circle-Ellipse problem 
    # (https://en.wikipedia.org/wiki/Circle%E2%80%93ellipse_problem)
    '''A spherical body with arbitrary radius and center'''
    def __init__(self,
        radius : float=1.0,
        center : np.ndarray[Shape[3], Numeric]=None,
    ) -> 'Sphere':
        super().__init__(
            radii=np.array([radius, radius, radius]),
            center=center,
        )
        self.radius = radius # propagate radus attribute to self
        
    def __repr__(self):
        return f'{self.__class__.__name__}(radius={self.radius})'
