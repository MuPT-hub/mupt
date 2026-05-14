'''Utilities for creating rigid transformations and applying them to points in 3D space
i.e. for working with the Special Euclidean group SE(3)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from .application import apply_rigid_transformation_recursive, RigidlyTransformable
from .rotations import rotator, rodrigues, alignment_rotation
from .alignment import rigid_vector_coalignment


def random_rigid_transformation(
    centered : bool=False,
    translation_bound : float=1.0,
) -> RigidTransform:
    """
    Generate a random rigid transformation, with both translation and rotation components by default

    Parameters
    ----------
    centered : bool, default False
        Whether the transformation should contain a translational component
    translation_bound : float, default 1.0
        Uniform bound along all Cartesian axes for translation component
        E.g. translation_bound=5.0 will pick a random translation vector from the set [-5, 5]**3

    Returns
    -------
    rand_transform : RigidTransform
        The resulting randomized rigid transformation
    """
    if centered:
        translation_bound = 0.0
    
    return RigidTransform.from_components(
        translation=np.random.uniform(-translation_bound, translation_bound, size=3),
        rotation=Rotation.random(),
    )