"""
Transformations from the more general affine group,
which allows scaling, origin shifts, and projections,
as well as utilities from converting to and from homogeneous coordinates.
"""

from .matrices import (
    AffineMatrix4x4 as AffineMatrix4x4,
    affine_matrix_from_linear_and_center as affine_matrix_from_linear_and_center,
    translation as translation,
    scaling as scaling,
    rotation_x as rotation_x,
    rotation_y as rotation_y,
    rotation_z as rotation_z,
    rotation_random as rotation_random,
)
from .homogeneous import (
    to_homogeneous_coords as to_homogeneous_coords,
    from_homogeneous_coords as from_homogeneous_coords,
)
from .application import (
    AffineTransformable as AffineTransformable,
    apply_affine_transformation_to_points as apply_affine_transformation_to_points,
    apply_affine_transformation_recursive as apply_affine_transformation_recursive,
)
