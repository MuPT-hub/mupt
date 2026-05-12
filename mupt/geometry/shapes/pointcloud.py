'''For representing clusters of positional coordinate, as one might find in a molecular conformer'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Iterable, Literal, Optional

from functools import cached_property
from itertools import product as cartesian

import numpy as np
from scipy.spatial.transform import RigidTransform
from scipy.spatial import ConvexHull, Delaunay

from .shapes import BoundedTransformableShape
from ..arraytypes import (
    Shape,
    NumberLike,
    Vector3,
    ArrayNx3,
    BitVectorN,
    TriangulationIndices,
)
from ...mutils.copyable import clear_cached_properties


class PointCloud(BoundedTransformableShape):
    '''A cluster of points in 3D space'''
    def __init__(self, positions : Optional[ArrayNx3]=None) -> None:
        if positions is None:
            positions = np.empty((0, 3), dtype=float)
        self.positions = np.atleast_2d(positions)

    def __repr__(self) -> str: 
        return f'{self.__class__.__name__}(shape={self.positions.shape})'
    
    @classmethod
    def cubic(cls, sidelen : float=1.0, centered : bool=True) -> 'PointCloud':
        '''
        Initialize a PointCloud whose point lie on the
        vertices of a cube with the given side lengths
        
        Parameters
        ----------
        sidelen : float, default 1.0
            The sidelen of the embedded cube
        centered : bool, default True
            Whether or not to center the vertices about the origin
            * If True, vertices will lie at the points (±S/2, ±S/2, ±S/2)
            * If False, the first vertex will lie at (0, 0, 0)
            and the rest extend by S into the first quadrant
        
        Returns
        -------
        cubic : PointCloud
            A PointCloud instance whose 8 vertices 
            are the vertices of the specified cube
        
        '''
        vertices : Iterable[tuple[int, int, int]] = cartesian(*[(0.0, sidelen) for _ in range(3)])
        positions : np.ndarray[Shape[Literal[8], Literal[3]], np.floating] = np.array(list(vertices))
        if centered:
            positions -= (sidelen / 2)

        return cls(positions)

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
    def centroid(self) -> Vector3:
        '''Geometric (i.e. unweighted) center of mass'''
        return self.positions.mean(axis=0)
    
    @property
    def volume(self) -> NumberLike:
        '''Volume of the convex hull of the positions in this PointCloud'''
        return self.convex_hull.volume
    
    def contains(self, points : Vector3 | ArrayNx3) -> BitVectorN:
        return np.atleast_1d(
            self.triangulation.find_simplex(points) != -1
        ).astype(object) # need to cast from numpy bool to Python bool

    def scale(self, scaling_factor : float) -> None:
        self.positions = scaling_factor*self.positions + (1 - scaling_factor)*self.centroid

    def surface_mesh(self) -> tuple[ArrayNx3, TriangulationIndices]:
        verts = self.convex_hull.vertices  # NB: self.convex_hull.points returns ALL points, even in interior 
        remap : dict[int, int] = {
            old_idx : new_idx
                for new_idx, old_idx in enumerate(verts)
        }
        remapped = np.vectorize(lambda x: remap.get(x, x))

        return self.positions[verts], remapped(self.convex_hull.simplices)
    
    # fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'PointCloud':
        return self.__class__(positions=np.array(self.positions))

    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        self.positions = transformation.apply(self.positions)
        clear_cached_properties(self) # invalidate cached qHull objects to prevent invariant plotting bug

    # derived quantities
    @property
    def bounding_radius(self) -> float:
        '''
        Bounding radius of the point cloud, defined as the 
        maximum center-of-mass distance scross all points
        '''
        return np.linalg.norm(self.positions - self.centroid, axis=1).max()
    r_bound = bounding_radius