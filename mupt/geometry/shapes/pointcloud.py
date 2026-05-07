'''For representing clusters of positional coordinate, as one might find in a molecular conformer'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Optional
from functools import cached_property

import numpy as np
from scipy.spatial.transform import RigidTransform
from scipy.spatial import ConvexHull, Delaunay

from .shapes import BoundedTransformableShape
from ..arraytypes import NumberLike, Vector3, ArrayNx3, TriangulationIndices
from ...mutils.copyable import clear_cached_properties


class PointCloud(BoundedTransformableShape):
    '''A cluster of points in 3D space'''
    def __init__(self, positions : Optional[ArrayNx3]=None) -> None:
        if positions is None:
            positions = np.empty((0, 3), dtype=float)
        self.positions = np.atleast_2d(positions)

    def __repr__(self) -> str: 
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
    def centroid(self) -> Vector3:
        '''Geometric (i.e. unweighted) center of mass'''
        return self.positions.mean(axis=0)
    
    @property
    def volume(self) -> NumberLike:
        '''Volume of the convex hull of the positions in this PointCloud'''
        return self.convex_hull.volume
    
    def contains(self, points : Vector3 | ArrayNx3) -> bool:
        return (self.triangulation.find_simplex(points) != -1).astype(object) # need to cast from numpy bool to Python bool

    def scale(self, scaling_factor : float) -> None:
        self.positions = scaling_factor*self.positions + (1 - scaling_factor)*self.centroid

    def surface_mesh(self) -> tuple[ArrayNx3, TriangulationIndices]:
        return self.convex_hull.points, self.convex_hull.simplices
    
    # fulfilling RigidlyTransformable contracts
    def _copy_untransformed(self) -> 'PointCloud':
        return self.__class__(positions=np.array(self.positions))

    def _rigidly_transform(self, transformation : RigidTransform) -> None:
        self.positions = transformation.apply(self.positions)
        clear_cached_properties(self) # invalidate cached qHull objects to prevent invariant plotting bug