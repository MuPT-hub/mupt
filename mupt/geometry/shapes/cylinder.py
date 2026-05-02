'''For representing cylindrical, rodlike bodies'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional

import numpy as np
from scipy.spatial.transform import RigidTransform

from .shapes import BoundedTransformableShape
from ..arraytypes import NumberLike, Vector3, ArrayNx3, TriangulationIndices
from ..measure import normalized


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
    
    def contains(self, points : Vector3 | ArrayNx3) -> bool:
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
        n_r, n_theta = 100, 200

        # # calculate cylinder wall coordinates
        # r, theta = np.mgrid[0.0:R:n_r*1j, 0.0:2*np.pi:n_theta*1j]
        # x_cyl = R * np.cos(theta) # fix radius for walls
        # y_cyl = R * np.sin(theta) # fix radius for walls
        # z, _ = np.mgrid[-L/2:L/2:n_r*1j, 0.0:2*np.pi:n_theta*1j]

        # # calculate face coordinates
        # x_face = r * np.cos(theta) # vary radius and fix Z for caps
        # y_face = r * np.sin(theta) # vary radius and fix Z for caps
        # z_face1 = np.full((n_r, n_theta), fill_value=-L/2)
        # z_face2 = np.full((n_r, n_theta), fill_value=L/2)

        # # stack XYZ coordinates together
        # cyl_pos_ref   = np.dstack([x_cyl, y_cyl, z])
        # face1_pos_ref = np.dstack([x_face, y_face, z_face1])
        # face2_pos_ref = np.dstack([x_face, y_face, z_face2])

        # apply transform to position in space
        ## TODO: triangulate + append face midpoints to triangulation
    
Rod = Cylinder # alias for convenience