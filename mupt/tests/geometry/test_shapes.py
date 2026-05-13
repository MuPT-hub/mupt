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


# DEV: deliberately not a fixture, since iterated over in parametrize args
def shapes() -> list[BoundedTransformableShape]: 
    '''Sample instances of untransformed BoundedTransformableShapes to test on'''
    return [
        PointCloud(np.random.rand(40,3)),
        Cylinder(1.2, 4),
        Sphere(3.5),
        Ellipsoid.from_components(1, 1, 2),
    ]

def shapes_transformed() -> list[BoundedTransformableShape]:
    '''Transformed versions of the sample test BoundedTransformableShape instances returned by `shapes()`'''
    return [
        shape.rigidly_transformed(random_rigid_transformation())
            for shape in shapes()
    ]

def shapes_mixed() -> list[BoundedTransformableShape]:
    '''Combined collection of transformed and untransformed example shapes'''
    return [*shapes(), *shapes_transformed()]

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
    # TODO: implement BoundedShape.__eq__() (and on subtypes) to permit easy comparisons
    ...

@pytest.mark.parametrize('shape,scaling_factor', cartesian(shapes(), (0.5, 1.0, 2.0)))
def test_volume_scaling(shape : BoundedShape, scaling_factor : float) -> None:
    '''Test that computed volume of shapes changes as expected with scaling'''
    v_orig : float = shape.volume
    v_scaled : float = shape.scaled(scaling_factor).volume

    nptest.assert_allclose(v_scaled / v_orig, scaling_factor**3)

@pytest.mark.parametrize('shape', shapes_mixed())
def test_containment_centroidal(shape : BoundedShape) -> None:
    '''Test that BoundedShapes contain their centroid''' 
    # DEV TB: assumes shapes are convex - true at time of writing, but may need to revisit in the future
    assert shape.contains(shape.centroid).all()

@pytest.mark.parametrize(
    'shape,scaling_factor,all_inside',
    [
        (shape, scaling_factor, all_inside)
            for shape, (scaling_factor, all_inside) in cartesian(
                shapes_mixed(),
                {
                    0.25 : True,
                    0.5  : True,
                    2.0  : False,
                    3.14 : False,
                }.items(),
            )
    ],
)
def test_containment_scaled(
    shape : BoundedTransformableShape,
    scaling_factor : float,
    all_inside : bool, 
) -> None:
    '''Test containment checks on shapes, relative to dilated and compressed versions of themselves'''
    # NB: in this SPECIFIC case, uniform scaling of convex shapes about center by non-unity scaling factor
    # means either the scaled copy contains the original (if factor >1) or vice-versa (if <1)
    mesh_points, triangles = shape.scaled(scaling_factor).surface_mesh() # implicitly also tests surface_mesh() - convenient, but not very atomic
    assert np.all(shape.contains(mesh_points) == all_inside)

@pytest.mark.parametrize(
    'shape,transformation,shape_transformed_expected',
    [],
)
def test_shape_rigidly_transformed(
    shape : BoundedTransformableShape,
    transformation : RigidTransform,
    shape_transformed_expected : BoundedTransformableShape,
) -> None:
    '''Test thatout-of-place rigid transformations of shapes give the expected output shape'''
    shape_init = shape.copy()
    shape_transformed = shape.rigidly_transform(transformation)
    
    # ensure target output is attained AND that original is unmodified
    assert (shape_transformed == shape_transformed_expected) and (shape == shape_init)