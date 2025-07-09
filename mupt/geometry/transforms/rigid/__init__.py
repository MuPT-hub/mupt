'''Utilities for creating rigid transformations and applying them to points in 3D space
i.e. for working with the Special Euclidean isometry group SE(3)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from .application import apply_rigid_transformation_recursive, RigidTransformable
from .rotations import rotator, rodrigues, alignment_rotation