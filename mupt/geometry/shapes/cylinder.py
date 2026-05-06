'''For representing cylindrical, rodlike bodies'''

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
    ArrayNx3,
    TriangulationIndices,
    BitVectorN,
)
from ..measure import normalized
from ..transforms.rigid.rotations import alignment_rotation


def cylindrical_mesh(
    radius : float,
    length : float,
    n_theta : int=30,
    n_z : int=5,
    direction : Optional[Vector3]=None,
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
    ...

    Returns
    -------
    ...
    '''
    # compute positions of mesh points
    params = zs, theta = np.mgrid[
        -length/2:length/2:n_z*1j,
        0.0:2*np.pi:(n_theta+1)*1j  # need +1 to get right number of polygon sides (since last is coincident with first)
    ]
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)

    mesh_points = np.dstack([xs, ys, zs]).reshape(-1, 3)
    mesh_points = np.concatenate([-cyl.axis[None, :], mesh_points, cyl.axis[None, :]]) # face midpoints are adjacent to runs of their neighbor points
    mesh_points = transformation.apply(mesh_points)
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
            axial_direction = np.array([0., 0., 1.]) # default to pointing in z-direction

        self.radius = radius
        self.length = length
        # DEV: opted to have the normal stored internally for 3 reasons:
        # 1) avoids need to rescale when scaling (accounted for by ".axis" property) 
        # 2) decreases likelihood of numerical instability when applying rigid transformations
        # 3) avoids needing to renormalize each time the axis is transformed
        self.axis_normal = normalized(axial_direction) 
        self.center = center
        self.cumulative_transformation *= RigidTransform.from_translation(center)

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

    def surface_mesh(self, n_theta : int=30, n_z : int=5) -> tuple[ArrayNx3, TriangulationIndices]:
        return cylindrical_mesh(
            self.radius,
            self.length,
            n_theta=n_theta,
            n_z=n_z,
            direction=self.axis_normal,
            transformation=self.cumulative_transformation,
        )
    
Rod = Cylinder # alias for convenience