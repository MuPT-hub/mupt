"""Unit tests for BoundedShape classes"""

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest
from itertools import product as cartesian

import numpy as np
import numpy.testing as nptest
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.geometry.transforms.rigid import random_rigid_transformation
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
        Cylinder(1.2, 4),
        Sphere(3.5),
        Ellipsoid.from_components(1, 1, 2),
    ]

def shapes_with_volumes() -> list[tuple[BoundedTransformableShape, float]]:
    '''Collection of shapes with known volumes, returned as (shape, expected volume) pairs'''
    return [
        (PointCloud.cubic(2.0), 8),
        (Cylinder(2, 4), 16*np.pi),
        (Sphere(3.0), 36*np.pi),
        (Ellipsoid.from_components(1, 2, 3), 8*np.pi),
    ]

@pytest.mark.parametrize('shape,volume_expected', shapes_with_volumes())
def test_volume(shape : BoundedShape, volume_expected : float) -> None:
    '''Test that volume calculation is accurate'''
    nptest.assert_allclose(shape.volume, volume_expected)

@pytest.mark.parametrize('shape,volume_expected', shapes_with_volumes())
def test_volume_transformed(shape : BoundedTransformableShape, volume_expected : float) -> None:
    '''Test that volume calculations remain invariant under rigid transformations'''
    shape_transformed = shape.rigidly_transformed(random_rigid_transformation())
    nptest.assert_allclose(shape_transformed.volume, volume_expected) # rigid motions have unit determinant and shouldn't affect volumes

@pytest.mark.parametrize('shape', shapes())
def test_scaling(shape : BoundedShape) -> None:
    ...

@pytest.mark.parametrize('shape,scaling_factor', cartesian(shapes(), (0.5, 1.0, 2.0)))
def test_volume_scaling(shape : BoundedShape, scaling_factor : float) -> None:
    '''Test that computed volume of shapes changes as expected with scaling'''
    v_orig : float = shape.volume
    v_scaled : float = shape.scaled(scaling_factor).volume

    nptest.assert_allclose(v_scaled / v_orig, scaling_factor**3)


@pytest.mark.parametrize('shape', shapes())
def test_containment(shape : BoundedShape) -> None:
    ...

@pytest.mark.parametrize('shape', shapes())
def test_rigid_transforms(shape : BoundedTransformableShape) -> None:
    ...