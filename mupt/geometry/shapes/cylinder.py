'''For representing cylindrical, rodlike bodies'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation, RigidTransform

from .shapes import BoundedTransformableShape
from ..arraytypes import (
    NumberLike,
    Vector3,
    ArrayNx3,
    TriangulationIndices,
    BitVectorN,
)
from ..measure import normalized
from ..transforms.rigid.rotations import alignment_rotation


Z_UNIT = np.array([0., 0., 1.])
Z_UNIT.setflags(write=False) # make immutable

def cylindrical_mesh(
    radius : float,
    length : float,
    n_theta : int=30,
    n_z : int=5,
    transformation : RigidTransform=RigidTransform.identity(),
) -> tuple[ArrayNx3, TriangulationIndices]:
    '''
    Generate a mesh of points defining the surface of a cylinder,
    including the walls and both faces.

    Without an applied transformation, will be centered at the origin
    with both faces a distance L/2 and -L/2, respectively, relative to
    the provided normal direction (Default z-axis)
    
    Parameters
    ----------
    radius : float
        The radius of the cylinder
    length : float
        The axial ("face-to-face") length of the cylinder
    n_theta : int, default 30
        Number of points to sample in the angular direction
        Will resemble an extruded regular n_theta-gon, e.g. n_theta=4 will be a square prism
    n_z : int, default 5
        Number of points to sample along the cylinder walls in the axial direction
        E.g. n_z=5 will yield a mesh with bands around the bottom
        face, 1/4 way up, 1/2 way up, 3/4 way up, and the top face
    transformation : RigidTransform, default RigidTransform.identity()
        A rigid transformation (e.g. combined rotation + translation) to apply to the cylinder
        Used to draw a cylinder which has been rotated and/or displaced from the origin

        By default, will apply the identity transformation, resulting in a cylinder
        parallel to the z-axis with centroid coincident with the origin
        i.e. with top and bottom faces at z=-L/2 and z=+L/2, respectively

    Returns
    -------
    mesh_points : ndarray[[P, 3], float]
        The points fo the 3D mesh on the surface of the cylinder
        
        The first point will be the midpoint of the bottom face, and the
        n_theta next points will be the band of neighboring points around the bottom face

        Likewise, the last point in the array will be the midpoint of the top face,
        with the preceding n_theta points being its neighbors around the edge of the top face

        NB: We take "top" here to mean the face in the axial direction,
        and "bottom" to mean the face in the opposite direction  
    triangles : ndarray[[T, 3], int]
        Array of the triples of indices defining triangular faces in the mesh 
    '''
    # compute positions of mesh points
    params = zs, theta = np.mgrid[
        -length/2:length/2:n_z*1j,
        0.0:2*np.pi:(n_theta+1)*1j  # need +1 to get right number of polygon sides (since last is coincident with first)
    ]
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    axis = length/2 * Z_UNIT

    mesh_points = np.dstack([xs, ys, zs]).reshape(-1, 3)
    mesh_points = np.concatenate([
        -axis[None, :],
        mesh_points,
        axis[None, :]]
    ) # face midpoints are adjacent to runs of their neighbor points
    mesh_points = transformation.apply(mesh_points) # axial tilts should be bundled here
    n_points = len(mesh_points)

    # triangulate mesh points
    ## bottom face
    triangles_face_bottom = np.zeros((n_theta, 3), dtype=int)
    face_bottom_edge_idxs = np.arange(1, n_theta + 1)
    triangles_face_bottom[:, 0] = 0
    triangles_face_bottom[:, 1] = face_bottom_edge_idxs
    triangles_face_bottom[:, 2] = np.roll(face_bottom_edge_idxs, -1)

    ## top face
    triangles_face_top = np.zeros((n_theta, 3), dtype=int)
    face_top_edge_idxs = np.arange(n_points - n_theta - 1, n_points - 1)
    triangles_face_top[:, 0] = n_points - 1
    triangles_face_top[:, 1] = face_top_edge_idxs
    triangles_face_top[:, 2] = np.roll(face_top_edge_idxs, -1)

    ## walls
    triangles_wall = Delaunay(params.reshape(2, -1).T).simplices + 1 # offset accounts for base point prepended to mesh positions
    triangles = np.concatenate([triangles_face_bottom, triangles_wall, triangles_face_top])

    return mesh_points, triangles


class Cylinder(BoundedTransformableShape):
    '''A cylindrical body with arbitrary radius, height, and center'''
    def __init__(
        self,
        radius : float=1.0,
        length : float=2.0,
        center : Optional[Vector3]=None,
        axial_direction : Optional[Vector3]=None,
    ) -> None:
        if center is None:
            center = np.zeros(3, dtype=float)
        center_std = np.atleast_2d(center).reshape(-1) # permits transposed and nested vector inputs
        assert center_std.shape == (3,)

        if axial_direction is None:
            axis_normal = Z_UNIT # default to pointing in z-direction
            axial_rotation = Rotation.identity()
        else:
            axis_normal = normalized(axial_direction.astype(float))
            axial_rotation = alignment_rotation(Z_UNIT, axis_normal)

        # DEV: opted to have the normal stored internally for 3 reasons:
        # 1) avoids need to rescale when scaling (accounted for by ".axis" property) 
        # 2) decreases likelihood of numerical instability when applying rigid transformations
        # 3) avoids needing to renormalize each time the axis is transformed
        self.radius = radius
        self.length = length
        self.axis_normal = axis_normal 
        self.center = center
        self.cumulative_transformation *= RigidTransform.from_components(
            translation=center,
            rotation=axial_rotation,
        )

    @classmethod
    def from_radius_and_axis(
        cls,
        radius : float,
        axis_vector : Vector3,
        center : Optional[Vector3]=None,
    ) -> 'Cylinder':
        '''
        Initialize Cylinder from axial vector (whose length is
        half the length of the cylinder), centroid, and radius
        '''
        return cls(
            radius=radius,
            length=np.linalg.norm(axis_vector),
            center=center,
            axial_direction=axis_vector,
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(radius={self.radius}, length={self.length})'
    
    @property
    def R(self) -> float:
        '''Alias for self.radius'''
        return self.radius

    @property
    def L(self) -> float:
        '''Alias for self.radius'''
        return self.length

    @property
    def axis(self) -> Vector3:
        '''Th vector spanning from the centroid to the center of the leading face'''
        return (self.length / 2) * self.axis_normal
    
    @property
    def face_centers(self) -> tuple[Vector3, Vector3]:
        '''The absolute positions of the midpoints of the leading and tailing faces on the cylinder'''
        return (self.center + self.axis, self.center - self.axis)

    # fulfilling BoundedShape contracts
    @property
    def centroid(self) -> Vector3:
        return self.center
    
    @property
    def volume(self) -> NumberLike:
        return np.pi * self.radius**2 * self.length
    
    def contains(self, points : Vector3 | ArrayNx3) -> BitVectorN:
        points_centered = np.atleast_2d(points - self.center)
         # double-transpose needed to get broadcast for multiplication right
        points_axial = (np.dot(points_centered, self.axis_normal) * points_centered.T).T
        points_radial = points_centered - points_axial

        within_axis = np.linalg.norm(points_radial, axis=1) <= (self.length / 2)
        within_radius = np.linalg.norm(points_axial, axis=1) <= self.radius

        return (within_axis & within_radius).astype(object)

    def scale(self, scaling_factor : float) -> None:
        self.radius *= scaling_factor
        self.length *= scaling_factor

    # fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'Cylinder':
        return self.__class__(
            radius=self.radius,
            length=self.length,
            center=np.array(self.center),
            axial_direction=np.array(self.axis_normal),
        )

    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        self.axis_normal = transformation.apply(self.axis_normal)
        self.center = transformation.apply(self.center)

    def surface_mesh(self, n_theta : int=30, n_z : int=5) -> tuple[ArrayNx3, TriangulationIndices]:
        return cylindrical_mesh(
            self.radius,
            self.length,
            n_theta=n_theta,
            n_z=n_z,
            transformation=self.cumulative_transformation,
        )
    
Rod = Cylinder # alias for convenience