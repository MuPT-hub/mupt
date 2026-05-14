'''For representing cylindrical, rodlike bodies'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Literal, Optional

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation, RigidTransform

from .shapes import BoundedTransformableShape
from ..arraytypes import (
    Shape,
    NumberLike,
    Vector3,
    ArrayNx3,
    TriangulationIndices,
    BitVectorN,
)
from ..measure import normalized, vector_flexible
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
    half_length : float = length / 2
    params = zs, theta = np.mgrid[
        -half_length:half_length:n_z*1j,
        0.0:2*np.pi:(n_theta+1)*1j  # need +1 to get right number of polygon sides (since last is coincident with first)
    ]
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    axis = half_length * Z_UNIT

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
        '''
        Parameters
        ----------
        radius : float, default 1.0
            The radius of the Cylinder, orthogonal to its central axis
        length : float, default 2.0
            The distance between the two parallel faces of the cylinder
        center : Optional[Vector3], default [0., 0., 0.]
            The absolute position of the geometric center of the cylinder
            If not explicitly provided, or provided as NoneType, will default to the origin, i.e. [0., 0., 0.]
        axial_direction : Optional[Vector3], default [0., 0., 1.]
            The direction from the center in which the leading ("top") face of the cylinder lies
            The length of this vector is inconsequential, and will be normalized to `length / 2`
            i.e. each face lies half of the Cylinder's length away from its center
            
            If not explicitly provided, or provided as NoneType, will default to the +z axis, i.e. [0., 0., 1.]
        '''
        if center is None:
            center = np.zeros(3, dtype=float)
        center = vector_flexible(center, dimension=3, dtype=float)

        half_length : float = length / 2
        if axial_direction is None:
            axis_vector = half_length * Z_UNIT # default to pointing in z-direction
            axial_rotation = Rotation.identity()
        else:
            axis_normal = normalized(axial_direction.astype(float))
            axial_rotation = alignment_rotation(Z_UNIT, axis_normal)
            axis_vector = half_length * axis_normal

        self.radius = radius
        self.length = length
        self.center = center
        # DEV TB: storing absolute positions, rather than relative axis vector,
        # to ensure cylinder changes as expected under rigid transformations
        # Axis vector is calculated as difference in-site
        self.face_center_top = center + axis_vector
        self.face_center_bottom = center - axis_vector
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
        Initialize Cylinder from a radius, axial vector, and centroid
        
        Parameters
        ----------
        radius : float
            The radius of the Cylinder, orthogonal to its central axis
        axial_vector : Vector3
            A vector whose direction is parallel to the center-to-face direction of the Cylinder
            and whose length is half of the intended length of the Cylinder
            
            E.g. axis_vector=np.array([0., 1., 0.,]) yields a Cylinder of length 2 parallel to the y-axis
        center : Optional[Vector3], default [0., 0., 0.]
            The absolute position of the geometric center of the Cylinder
            If not explicitly provided, or provided as NoneType, will default to the origin, i.e. [0., 0., 0.]
        
        Returns
        -------
        cylinder : Cylinder
            A cylinder instance pointing in the desired direction
        '''
        return cls(
            radius=radius,
            length=2*np.linalg.norm(axis_vector),
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
        '''
        The vector spanning from the centroid to the center of the leading face
        Has length equal to half the length of the Cylinder
        '''
        return self.face_center_top - self.center
    axis_vector = axis
    
    @property
    def axis_normal(self) -> Vector3:
        '''Unit vector in the direction from the centroid to the center of the leading face'''
        return normalized(self.axis)
    
    @property
    def face_centers(self) -> np.ndarray[
        Shape[Literal[3], Literal[2]],
        np.dtype[np.floating],
    ]:
        '''The absolute positions of the midpoints of the leading and tailing faces on the cylinder'''
        return np.vstack((self.face_center_top, self.face_center_bottom))

    # fulfilling BoundedShape contracts
    @property
    def centroid(self) -> Vector3:
        return self.center
    
    @property
    def volume(self) -> NumberLike:
        return np.pi * self.radius**2 * self.length
    
    def contains(self, points : Vector3 | ArrayNx3) -> BitVectorN:
        points_centered = np.atleast_2d(points - self.center)
        points_axial = np.outer(np.dot(points_centered, self.axis_normal), self.axis_normal)
        points_radial = points_centered - points_axial

        within_axis = np.linalg.norm(points_radial, axis=1) <= (self.length / 2)
        within_radius = np.linalg.norm(points_axial, axis=1) <= self.radius

        return (within_axis & within_radius).astype(object)

    def congruent_to(self, other : 'Cylinder') -> bool:
        return (self.radius == other.radius) \
            and (self.length == other.length) \
            and np.allclose(self.center, other.center)

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
        self.center = transformation.apply(self.center)
        self.face_center_top = transformation.apply(self.face_center_top)
        self.face_center_bottom = transformation.apply(self.face_center_top)
        print(np.linalg.norm(self.axis_normal))

    def surface_mesh(self, n_theta : int=30, n_z : int=5) -> tuple[ArrayNx3, TriangulationIndices]:
        return cylindrical_mesh(
            self.radius,
            self.length,
            n_theta=n_theta,
            n_z=n_z,
            transformation=self.cumulative_transformation,
        )
    
Rod = Cylinder # alias for convenience