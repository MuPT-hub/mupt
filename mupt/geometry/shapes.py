'''For encoding rigid bodies in space'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generic, Optional, Sequence, Union

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation, RigidTransform

from .arraytypes import Shape, Numeric, M, N, P
from .coordinates.basis import (
    is_columnspace_mutually_orthogonal,
    is_orthogonal,
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
    
    def contains(self, *point : Sequence[Numeric]) -> bool:
        '''Test whether a point lies on the plane defined'''
        if len(point) == 1 and isinstance(point[0], (Sequence, np.ndarray)):
            point = point[0] # correct missing star-args for Sequence-like
            # TODO: convert Sequences to numpy arrays
        
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
        
    @abstractmethod
    def apply_rigid_transformation(self, transform : RigidTransform) -> 'BoundedShape':
        '''Apply a rigid transformation to the body'''
        ...
     
    # @abstractmethod
    # def support(self, direction : np.ndarray[Shape[3], Numeric]) -> np.ndarray[Shape[3], Numeric]:
    #     '''Determines the furthest point on the surface of the body in a given direction'''
    #     ...
        

# Concrete BoundedShape implementations
class PointCloud(BoundedShape[Numeric]):
    '''A cluster of points in 3D space'''
    def __init__(self, positions : np.ndarray[Shape[3], Numeric]=None) -> None:
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
    
    @property
    def centroid(self) -> np.ndarray[Shape[3], Numeric]:
        return self.positions.mean(axis=0)
    
    @property
    def volume(self) -> Numeric:
        return self.convex_hull.volume
    
    def contains(self, points : np.ndarray[Union[Shape[3], Shape[N, 3]]]) -> bool:
        return (self.triangulation.find_simplex(points) != -1).astype(object) # need to cast from numpy bool to Python bool

    def apply_rigid_transformation(self, transform : RigidTransform) -> 'PointCloud':
        return PointCloud(positions=transform.apply(self.positions))
    
@dataclass
class Ellipsoid(BoundedShape[Numeric]):
    '''
    A generalized spherical body, with potentially asymmetric orthogonal principal axes and arbitrary centroid
    
    Represented by a (not necessarily isotropic) scaling of the basis vectors and a rigid transformation,
    which, together, map the points on a unit sphere at the origin to the surface of the ellipsoid
    '''
    radii  : np.ndarray[Shape[3], Numeric] = field(default_factory=lambda: np.ones(3, dtype=float))  # by default, make a unit sphere
    center : np.ndarray[Shape[3], Numeric] = field(default_factory=lambda: np.zeros(3, dtype=float)) # by default, center at origin
    orientation : Rotation = field(default_factory=Rotation.identity) # by default, no rotation
    
    # TODO: check shapes of radii and center post-init
    
    def __eq__(self, other : 'Ellipsoid') -> bool:
        return np.allclose(self.radii, other.radii) \
            and np.allclose(self.center, other.center) \
            and np.allclose(self.orientation.as_matrix(), other.orientation.as_matrix())

    # initialization
    @classmethod
    def from_components(
        cls,
        radius_x : float=1.0,
        radius_y : float=1.0,
        radius_z : float=1.0, 
        center : Optional[np.ndarray[Shape[3], Numeric]]=None,
        orientation : Optional[Union[Rotation, np.ndarray[Shape[3, 3], Numeric]]]=None
    ) -> 'Ellipsoid':
        '''Instantiate an ellipsoid its axis lengths, center, and orientation in a more flexible format'''
        if center is None:
            center = np.zeros(3, dtype=float)
            
        if orientation is None:
            orientation = Rotation.identity()
        elif isinstance(orientation, np.ndarray):
            assert orientation.shape == (3, 3), 'Orientation must be a 3x3 rotation matrix' # TODO: make thse proper Exceptions
            assert is_orthogonal(orientation) , 'Orientation must be an orthogonal matrix'  # TODO: make thse proper Exceptions
            orientation = Rotation.from_matrix(orientation)
            
        return cls(
            radii=np.array([radius_x, radius_y, radius_z], dtype=float),
            center=center,
            orientation=orientation,
        )
        
    @classmethod
    def from_axis_lengths_and_transform(
        cls,
        radius_x : float=1.0,
        radius_y : float=1.0,
        radius_z : float=1.0, 
        transform : Optional[RigidTransform]=None,
    ) -> 'Ellipsoid':
        '''Instantiate an ellipsoid from its axis lengths and a rigid transformation'''
        return cls.from_components(
            radius_x=radius_x,
            radius_y=radius_y,
            radius_z=radius_z,
            center=None if (transform is None) else transform.translation,
            orientation=None if (transform is None) else transform.rotation,
        )

    # Matrix presentation of the ellipsoid
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
        
    @property
    def transformation(self) -> RigidTransform:
        '''The rigid transformation defining this ellipsoids center and orientation'''
        return RigidTransform.from_components(
            translation=self.center,
            rotation=self.orientation,
        )

    def scaling_matrix(self, as_affine : bool=True) -> np.ndarray[Union[Shape[3, 3], Shape[4, 4]], Numeric]:
        '''The scaling matrix which defines the radii of the ellipsoid'''
        if as_affine:
            return np.diag([*self.radii, 1.0])  # add a 1.0 for the homogeneous coordinate
        return np.diag(self.radii)
        
    @property
    def matrix(self) -> np.ndarray[Shape[4, 4], Numeric]:
        '''
        An affine matrix which represents this Ellipsoid
        
        Has the effect of transforming the unit sphere at the origin, 
        (in homogeneous coordinates) to the surface of this Ellipsoid
        '''
        return self.transformation.as_matrix() @ self.scaling_matrix(as_affine=True)
    basis = matrix
        
    @property
    def inverse(self) -> np.ndarray[Shape[4, 4], Numeric]:
        '''
        Transformation which maps this ellipsoid to the unit sphere centered at the origin
        Inverse of Ellipsoid.matrix
        '''
        return np.linalg.inv(self.matrix) # precompute inverse for later use
    inv = inverse
        
    # fulfilling BoundedShape contracts
    @property
    def centroid(self) -> np.ndarray[Shape[3], Numeric]:
        return self.center
    
    @property
    def volume(self) -> Numeric:
        return 4/3 * np.pi * np.prod(self.radii) # DEVNOTE: determination of rotation is always 1, so we may as well skip it and the whole determinant calculation
        # return 4/3 * np.pi * np.linalg.det(self.matrix)

    def contains(self, points : np.ndarray[Union[Shape[3], Shape[N, 3]]]) -> bool:   # TODO: decide whether containment should be boundary-inclusive
        return (np.linalg.norm(
            (self.transformation.inv().apply(points) / self.radii), # reduce containment check to comparison with auxiliary unit sphere
            axis=1,
        ) < 1).astype(object) # need to cast from numpy bool to Python bool

    def apply_rigid_transformation(self, transformation : RigidTransform) -> 'Ellipsoid':
        net_transformation = transformation * self.transformation
        return Ellipsoid(
            radii=self.radii,
            center=net_transformation.translation,
            orientation=net_transformation.rotation,
        )
    
    # visualization   
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

        a, b, c = self.radii 
        positions = np.zeros((n_theta, n_phi, 3), dtype=float)
        positions[..., 0] = a * r * np.sin(phi) * np.cos(theta)
        positions[..., 1] = b * r * np.sin(phi) * np.sin(theta)
        positions[..., 2] = c * r * np.cos(phi)
        
        return self.transformation.apply(
            positions.reshape(-1, 3) # flatten into (n_theta*n_phi)x3 array to allow RigidTransform.apply() to digest it
        ).reshape(n_theta, n_phi, 3) # ...then repackage into mesh for convenient plotting
    
    
class Sphere(Ellipsoid[Numeric]): # TODO: reimplement as separate from Ellipsoid
    # TODO: address creation from axes (https://en.wikipedia.org/wiki/Circle%E2%80%93ellipse_problem)
    '''A spherical body with arbitrary radius and center'''
    def __init__(self,
        radius : float=1.0,
        center : np.ndarray[Shape[3], Numeric]=None,
        orientation : Rotation=Rotation.identity(),
    ) -> 'Sphere':
        super().__init__(
            radii=np.array([radius, radius, radius]),
            center=center,
            orientation=orientation,
        )
        self.radius = radius
        
    def __repr__(self):
        return f'Sphere(r={self.radius})'
    
    # NOTE: affine transformations will produce Ellipsoid instances, as one would expect
