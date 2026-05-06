"""Unit tests for BoundedShape classes"""

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest 

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.geometry.shapes import (
    visualize_shape,
    BoundedShape,
    BoundedTransformableShape,
    PointCloud,
    Cylinder,
    Sphere,
    Ellipsoid,
)


# DEV: deliverately not a fixture, since iterated over in parametrize args
def shapes() -> list[BoundedTransformableShape]: 
    '''Sample instances of BoundedTransformableShapes to test on'''
    return [
        PointCloud(np.random.rand(40,3)),
        Cylinder(1, 4),
        Sphere(1),
        Ellipsoid.from_components(1, 1, 2),
    ]

@pytest.mark.parametrize('shape', shapes())
def test_volume(shape : BoundedShape) -> None:
    ...

@pytest.mark.parametrize('shape', shapes())
def test_scaling(shape : BoundedShape) -> None:
    ...

@pytest.mark.parametrize('shape', shapes())
def test_containment(shape : BoundedShape) -> None:
    ...

@pytest.mark.parametrize('shape', shapes())
def test_rigid_transforms(shape : BoundedTransformableShape) -> None:
    ...