"""Unit tests for BoundedShape classes"""

import pytest
from itertools import product as cartesian

import numpy as np
import numpy.testing as nptest
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.geometry.transforms.rigid import random_rigid_transformation
from mupt.geometry.shapes import (
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
        shape.rigidly_transformed(random_rigid_transformation(translation_bound=1.0))
            for shape in shapes()
    ]

def shapes_mixed() -> list[BoundedTransformableShape]:
    '''Combined collection of transformed and untransformed example shapes'''
    return [*shapes(), *shapes_transformed()]

def shapes_with_volumes() -> list[tuple[BoundedTransformableShape, float]]: 
    '''Collection of shapes with known volumes, returned as (shape, expected volume) pairs'''
    # DEV: only made function because multiple tests use these inputs
    return [
        (PointCloud.cubic(2.0), 8),
        (Cylinder(2, 4), 16*np.pi),
        (Sphere(3.0), 36*np.pi),
        (Ellipsoid.from_components(1, 2, 3), 8*np.pi),
    ]
    

# "Pure" shape tests
@pytest.mark.parametrize(
    'shape,other,expected_equal',
    [
        (PointCloud.cubic(2.0), PointCloud.cubic(2.0), True),
        (PointCloud.cubic(2.0), PointCloud(np.random.rand(8,3)), False),
        (Ellipsoid.from_components(1,2,3), Ellipsoid(radii=np.array([1.,2.,3.])), True),
        (Sphere(radius=1), Sphere(radius=1.0), True),
        (
            Cylinder(1.5, length=2*np.sqrt(3), axial_direction=np.array([1,1,1])),
            Cylinder.from_radius_and_axis(1.5, axis_vector=np.array([1,1,1])),
            True,
        ),
        (Sphere(radius=1), Cylinder(1, 1), False),
        (PointCloud.cubic(2.0), Sphere(3.14), False),
    ]
)
def test_comparison(
    shape : BoundedShape,
    other : BoundedShape,
    expected_equal : bool,
) -> None:
    '''Test that __eq__ is able to discern shape instances as expected'''
    assert (shape == other) == expected_equal

@pytest.mark.parametrize('shape,volume_expected', shapes_with_volumes())
def test_volume(shape : BoundedShape, volume_expected : float) -> None:
    '''Test that volume calculation is accurate'''
    nptest.assert_allclose(shape.volume, volume_expected)

@pytest.mark.parametrize(
    'shape,scaling_factor,shape_scaled_expected',
    [
        (PointCloud.cubic(1), 0.5, PointCloud.cubic(0.5)),
        (PointCloud.cubic(1), 3.0, PointCloud.cubic(3.0)),
        (Sphere(1), 0.5, Sphere(0.5)),
        (Sphere(1), 2.5, Sphere(2.5)),
        (Ellipsoid.from_components(1, 2, 3), 2.0, Ellipsoid.from_components(2, 4, 6)),
        (
            Ellipsoid(
                radii=np.array([4, 5, 6]),
                center=np.array([1., 2., 1.,])
            ),
            0.25, 
            Ellipsoid(
                radii=np.array([1.0, 1.25, 1.5]),
                center=np.array([1., 2., 1.,])
            ),
        ),
        (Cylinder(3.14, length=0.58), 0.75, Cylinder(2.355, length=0.435)),
        (
            Cylinder.from_radius_and_axis(3.6, np.array([1, 1, 1])),
            2.0,
            Cylinder.from_radius_and_axis(7.2, np.array([2, 2, 2])),
        ),
    ]
)    
def test_scaling(
    shape : BoundedShape,
    scaling_factor : float,
    shape_scaled_expected : BoundedShape,
) -> None:
    '''Test that scaling operations give the expected shape of the same type'''
    # scaled in-place, both to test that this also works AND because the BoundedShape
    # base does not implement .scaled() (only BoundedTranformableShape does this)
    shape.scale(scaling_factor) 
    assert shape == shape_scaled_expected

@pytest.mark.parametrize('shape,scaling_factor', cartesian(shapes(), (0.5, 1.0, 2.0)))
def test_volume_scaling(shape : BoundedShape, scaling_factor : float) -> None:
    '''Test that computed volume of shapes changes as expected with scaling'''
    v_orig : float = shape.volume
    v_scaled : float = shape.scaled(scaling_factor).volume

    nptest.assert_allclose(v_scaled / v_orig, scaling_factor**3)

@pytest.mark.parametrize('shape', shapes_mixed())
def test_containment_centroidal(shape : BoundedShape) -> None:
    '''Test that BoundedShapes contain their centroid''' 
    # DEV TB: assumes shapes are convex - true at time,
    # of writing but may need to revisit in the future
    assert shape.contains(shape.centroid).all()


# Transformable shape tests
@pytest.mark.parametrize('shape', shapes())
def test_equality(shape : BoundedTransformableShape) -> None:
    # NB: not checking direct equality with self, since that would be
    # true even if no __eq__ was implemented! (defaults to "is" behavior)
    assert shape.copy() == shape

@pytest.mark.parametrize('shape,volume_expected', shapes_with_volumes())
def test_volume_transformed(shape : BoundedTransformableShape, volume_expected : float) -> None:
    '''Test that volume calculations remain invariant under rigid transformations'''
    shape_transformed = shape.rigidly_transformed(random_rigid_transformation(translation_bound=1.0))
    nptest.assert_allclose(shape_transformed.volume, volume_expected) # rigid motions have unit determinant and shouldn't affect volumes

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
