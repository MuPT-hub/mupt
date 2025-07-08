'''
Transformations from the more general affine group, which allows scaling, origin shifts, and projections,
as well as utilities from converting to and from homogeneous coordinates.
'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from .matrices import (
    AffineMatrix4x4,
    affine_matrix_from_linear_and_center,
    translation,
    scaling,
    rotation_x,
    rotation_y,
    rotation_z,
    rotation_random,
)
from.homogeneous import (
    to_homogeneous_coords,
    from_homogeneous_coords,
)
from .application import (
    AffineTransformable,
    apply_affine_transformation_to_points,
    apply_affine_transformation_recursive,
)