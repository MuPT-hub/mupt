'''For representing ellipsoidal shapes, including spheres as a special case'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import RigidTransform

from .shapes import BoundedTransformableShape
from ..arraytypes import (
    NumberLike,
    Vector3,
    Array3x3,
    Array4x4,
    ArrayNxN,
    ArrayNx3,
    TriangulationIndices,
    BitVectorN,
)
from ..measure import vector_flexible
from ..coordinates.basis import is_columnspace_mutually_orthogonal

        
def ellipsoidal_mesh(
    rx : float,
    ry : Optional[float]=None,
    rz : Optional[float]=None,
    n_theta : int=30,
    n_phi : int=30,
    transformation : RigidTransform=RigidTransform.identity(),
) -> tuple[ArrayNx3, TriangulationIndices]:
    ''' 
    Generate a mesh of points defining the surface of an ellipsoid 
    (i.e. generalized sphere with 3 independent, arbitrarily sized readii)
    
    Parameters
    ----------
    rx : float
        Radius of the ellipsoid in the x-direction or,
        if no other radii are provided, radius of the sphere
    ry : Optional[float], default rx
        Radius of the ellipsoid in the y-direction
        If not explicitly provided, takes on the value assigned to rx
    rz : Optional[float], default rx
        Radius of the ellipsoid in the z-direction
        If not explicitly provided, takes on the value assigned to rx
    n_theta : int, default 30
        Number of points in the azimuthal angle direction
        Equivalent to longitudinal resolution
        
        Theta is taken to be the angle CC from the +x axis in the xy-plane,
        following the mathematics (not physics!) convention
    n_phi : int, default 30
        Number of points in the polar angle direction
        Equivalent to latitudinal resolution
        
        Phi is taken to be the angle "downwards" from the +z axis
        following the mathematics (not physics!) convention
    transformation : RigidTransform, default RigidTransform.identity()
        An additional rigid transformation (i.e. rotation + translation) 
        to apply to the ellipsoid from it's defined reference position
        
        Defaults to the identity transformation, returning the unmodified mesh points
        
    Returns
    -------
    mesh_points : ndarray[[MxP, 3], float]
        The points fo the 3D mesh on the surface of the cylinder
        M is the number of points in the azimuthal direction
        P is the number of points in the polar direction
    triangles : ndarray[[T, 3], int]
        Array of the triples of indices defining triangular faces in the mesh 
    '''
    if ry is None:
        ry = rx
    if rz is None:
        rz = ry

    angles = theta, phi = np.mgrid[
        0.0:2*np.pi:n_theta*1j,
        0.0:np.pi:n_phi*1j,
    ] # (magnitude of) complex step size is interpreted by numpy as a number of points
    triangulation = Delaunay(angles.reshape(2, -1).T) # note: .reshape(-1, 2) gives the right shape but NOT the right parity between parametric angles

    mesh_points = np.zeros((n_theta, n_phi, 3), dtype=float) # TODO: rewrite as dstack?
    mesh_points[..., 0] = rx * np.sin(phi) * np.cos(theta)
    mesh_points[..., 1] = ry * np.sin(phi) * np.sin(theta)
    mesh_points[..., 2] = rz * np.cos(phi)
    
    mesh_points = mesh_points.reshape(-1, 3) # flatten into (n_theta*n_phi)x3 array of XYZ positions
    mesh_points = transformation.apply(mesh_points) # apply transform

    return mesh_points, triangulation.simplices

class Sphere(BoundedTransformableShape): # N.B: doesn't inherit from Ellipsoid to avoid Circle-Ellipse problem (https://en.wikipedia.org/wiki/Circle%E2%80%93ellipse_problem)
    '''A spherical body with arbitrary radius and center'''
    def __init__(
        self,
        radius : float=1.0,
        center : Optional[Vector3]=None,
    ) -> None:
        if center is None:
            center = np.zeros(3, dtype=float)
        center = vector_flexible(center, dimension=3, dtype=float)

        self.radius = radius
        self.center = center
        self.cumulative_transformation *= RigidTransform.from_translation(center)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(radius={self.radius})'
    
    @property
    def R(self) -> float:
        '''Alias for self.radius'''
        return self.radius

    # fulfilling BoundedShape contracts
    @property
    def centroid(self) -> Vector3:
        return self.center

    @property
    def volume(self) -> float:
        return 4/3 * np.pi * self.radius**3
    
    def contains(self, points : Vector3 | ArrayNxN) -> BitVectorN:
        return (
            np.linalg.norm(
                np.atleast_2d(points - self.center),
                axis=1, # TODO: 
            ) <= self.radius
        ).astype(object)

    def congruent_to(self, other : 'Sphere') -> bool:
        return (self.radius == other.radius) \
            and np.allclose(self.center, other.center)
    
    def scale(self, scaling_factor : float) -> None:
        self.radius *= scaling_factor

    def surface_mesh(self, n_theta : int=30, n_phi : int=30) -> tuple[ArrayNx3, TriangulationIndices]:
        return ellipsoidal_mesh(
            rx=self.radius,
            # ry, rz forced to match rx if not passed explicitly
            n_theta=n_theta,
            n_phi=n_phi,
            transformation=self.cumulative_transformation,
        )
    
    # fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'Sphere':
        return self.__class__(
            radius=self.radius,
            center=np.array(self.center),
        )

    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        self.center = transformation.apply(self.center)
    
class Ellipsoid(BoundedTransformableShape):
    '''
    A generalized spherical body, with potentially asymmetric orthogonal principal axes and arbitrary centroid
    
    Representable by a (not necessarily isotropic) scaling of the basis vectors and a rigid transformation,
    which, together, map the points on a unit sphere at the origin to the surface of the Ellipsoid
    '''
    def __init__(
        self,
        radii  : Optional[Vector3]=None,
        center : Optional[Vector3]=None,
    ) -> None:
        # DEV: extract this vector shape checking into external utility, eventually
        if radii is None:
            radii = np.ones(3, dtype=float)
        radii = vector_flexible(radii, dimension=3, dtype=float)
            
        if center is None:
            center = np.zeros(3, dtype=float)
        center = vector_flexible(center, dimension=3, dtype=float)

        self.radii = radii
        self.center = center
        self.cumulative_transformation *= RigidTransform.from_translation(center)

    @classmethod
    def from_components(
        cls,
        # axis lengths
        radius_x : NumberLike=1.0,
        radius_y : NumberLike=1.0,
        radius_z : NumberLike=1.0,
        center_x : NumberLike=0.0,
        center_y : NumberLike=0.0,
        center_z : NumberLike=0.0,
        # center coordinate
    ) -> 'Ellipsoid':
        '''Instantiate Ellipsoid from array-wise representations of its radii and center'''
        return cls(
            radii=np.array([radius_x, radius_y, radius_z], dtype=float),
            center=np.array([center_x, center_y, center_z], dtype=float),
        )

    def __repr__(self) -> str: 
        return f'{self.__class__.__name__}(radii={self.radii}, center={self.center})'

    # Matrix representations of the Ellipsoid
    @staticmethod
    def is_valid_ellipsoid_matrix(basis : Array4x4) -> bool:
        '''Check that an affine matrix could represent an Ellipsoid'''
        assert basis.shape == (4, 4)
        # TODO: find more elegant way to do this splitting
        axes = basis[:-1, :-1]
        center = basis[:-1, -1]
        projective_part = basis[-1, :-1]
        w = basis[-1, -1]
        
        return bool(
            is_columnspace_mutually_orthogonal(axes) # ensure principal axes are mutually orthogonal
            and np.allclose(projective_part, 0.0) # ensure axes have apply no projective transformation
            and np.isclose(w, 1.0), # ensure homogeneous scale of the center is 1 (i.e. unprojected)
        )
        
    def scaling_matrix(self, as_affine : bool=True) -> Array3x3 | Array4x4:
        '''The scaling matrix which defines the radii of the Ellipsoid'''
        if as_affine:
            return np.diag([*self.radii, 1.0])  # add a 1.0 for the homogeneous coordinate
        return np.diag(self.radii)
        
    def as_affine_matrix(self) -> Array4x4:
        '''
        An affine matrix which represents this Ellipsoid
        
        Has the effect of transforming the unit sphere at the origin, 
        (in homogeneous coordinates) to the surface of this Ellipsoid
        '''
        return self.cumulative_transformation.as_matrix() @ self.scaling_matrix(as_affine=True)
    
    @property
    def basis(self) -> Array4x4:
        '''The basis matrix of the Ellipsoid - alias for Ellipsoid.as_affine_matrix()'''
        return self.as_affine_matrix()
    
    @property
    def principal_axes(self) -> Array3x3:
        '''The principal axes of the ellipsoid, represented as a 3x3 matrix
        whose rows are the axis vectors emanating from the Ellipsoid's center'''
        return self.cumulative_transformation.apply(self.scaling_matrix(as_affine=False))
    axes = principal_axes # alias

    def affine_inverse(self) -> Array4x4:
        '''
        Transformation which maps this Ellipsoid to the unit sphere centered at the origin
        Inverse of the Ellipsoid's affine basis matrix
        '''
        return np.linalg.inv(self.as_affine_matrix) # precompute inverse for later use
    
    @property
    def inv(self) -> Array4x4:
        '''
        The inverse of the Ellipsoid's affine basis matrix - alias for Ellipsoid.affine_inverse()
        Maps this Ellipsoid to the unit sphere centered at the origin
        '''
        return self.affine_inverse()

    def coincident_with(self, other : 'Ellipsoid') -> bool: # TODO: replace with __eq__
        return np.allclose(self.radii, other.radii) \
            and np.allclose(self.center, other.center) \
            and np.allclose(
                self.cumulative_transformation.as_matrix(),
                other.cumulative_transformation.as_matrix(),
            )
        
    # fulfilling BoundedShape contracts
    @property
    def centroid(self) -> Vector3:
        return self.center
    
    @property
    def volume(self) -> NumberLike:
        # return 4/3 * np.pi * np.linalg.det(self.matrix)
        return 4/3 * np.pi * np.prod(self.radii) # DEVNOTE: determinant of rotation is always 1, so we may as well skip it

    def contains(self, points : Vector3 | ArrayNxN) -> BitVectorN:
        # Reduce containment check to comparison with auxiliary unit sphere
        # NB: not applying self.inverse to points because the Ellipsoid basis
        # matrix in general not a rigid transformation because of axial stretching
        return ( 
            np.linalg.norm( 
                np.atleast_2d(self.resetting_transformation.apply(points) / self.radii), 
                axis=1,
            ) <= 1
        ).astype(object) # need to cast from numpy bool to Python bool
    
    def congruent_to(self, other : 'Sphere') -> bool:
        return np.allclose(self.radii, other.radii) \
            and np.allclose(self.center, other.center)

    def scale(self, scaling_factor : float) -> None:
        self.radii *= scaling_factor

    def surface_mesh(self, n_theta : int=30, n_phi : int=30) -> tuple[ArrayNx3, TriangulationIndices]:
        return ellipsoidal_mesh(
            *self.radii,
            n_theta=n_theta,
            n_phi=n_phi,
            transformation=self.cumulative_transformation,
        )

    # fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'Ellipsoid':
        return self.__class__(
            radii=np.array(self.radii),
            center=np.array(self.center),
        )

    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        self.center = transformation.apply(self.center)
