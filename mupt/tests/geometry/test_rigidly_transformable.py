'''Test that Protocol for RigidlyTransformable objects is implemented correctly'''

import pytest
from typing import Union
from itertools import product as cartesian

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.mutils.copyable import NotCopyableError
from mupt.geometry.transforms.rigid.application import RigidlyTransformable


# dummy classes for testing over
class Points(RigidlyTransformable):
    '''Dummy class for testing RigidlyTransformable Protocol'''
    def __init__(self, positions: np.ndarray):
        self.positions = positions

    def _rigidly_transform(self, transform: RigidTransform) -> None:
        self.positions = transform.apply(self.positions)
        
    def _copy_untransformed(self) -> 'Points':
        return self.__class__(positions=np.array(self.positions))
    
class PointsNonCopyable(RigidlyTransformable):
    '''Dummy class for to test that in-place methods fail when copying is undefined'''
    def __init__(self, positions: np.ndarray):
        self.positions = positions

    def _rigidly_transform(self, transform: RigidTransform) -> None:
        self.positions = transform.apply(self.positions)

    # NOTE: _copy_untransformed() method deliberately omitted
    
# Fixtures
@pytest.fixture(scope='function')
def transform() -> RigidTransform:
    direction = np.array([0.0, 1.0, 0.0])
    return RigidTransform.from_components(
        rotation=Rotation.from_rotvec(np.pi/4 * direction), # rotate an eighth of a turn clockwise about the target direction
        translation=direction, # slide 1 unit along the target direction
    )
    
@pytest.fixture(scope='function')
def sample_positions() -> np.ndarray:
    return np.array(list(cartesian([0.0, 1.0], repeat=3)), dtype=float)

@pytest.fixture(scope='function')
def sample_positions_transformed(transform: RigidTransform, sample_positions: np.ndarray) -> np.ndarray:
    return transform.apply(sample_positions)

@pytest.fixture(scope='function')
def points(sample_positions : np.ndarray) -> Points:
    return Points(positions=sample_positions)

@pytest.fixture(scope='function')
def points_non_copyable(sample_positions : np.ndarray) -> PointsNonCopyable:
    return PointsNonCopyable(positions=sample_positions)

# Tests
## Test in-place methods
def test_rigidly_transform(points : Points, transform : RigidTransform, sample_positions_transformed : np.ndarray):
    '''Test that rigid transformation are correctly applied in-place'''
    points.rigidly_transform(transform)
    np.testing.assert_allclose(
        points.positions,
        sample_positions_transformed,
        strict=True,
    )

@pytest.mark.parametrize('num_applications', range(8))
def test_cumulative_transformation(
        points : Points,
        transform : RigidTransform,
        num_applications : int,
    ):
    '''Test that transformed applied one after the other are correctly accumulated'''
    cumul_trans = RigidTransform.identity()
    for _ in range(num_applications):
        points.rigidly_transform(transform)
        cumul_trans *= transform
        
    np.testing.assert_allclose(
        points.cumulative_transformation.as_matrix(), # NOTE: must compare as matrix, as __eq__ does not perform this comparison
        cumul_trans.as_matrix(),
        strict=True, # check dtype and shape for completeness
    )

def test_reset_transform(points : Points, transform : RigidTransform):
    '''Test that resetting a rigid transformation in-place return the object to its original state'''
    points.rigidly_transform(transform)
    points.reset_transform()
    np.testing.assert_allclose(
        points.cumulative_transformation.as_matrix(),
        RigidTransform.identity().as_matrix(),
        strict=True,
    )

## Test read-only variants of methods
def test_rigidly_transformed(
    points : Union[Points, PointsNonCopyable],
    transform : RigidTransform,
):
    '''Test that rigid transformation are correctly applied in-place'''
    new_points = points.rigidly_transformed(transform)
    # NOTE: will not compare as expected when transform is within float imprecision
    # Also, not opting for np.testing functionality, since there is no direct support built in for checking arrays are NOT equal
    assert not np.allclose(new_points.positions, points.positions)

@pytest.mark.xfail(
    reason='Can\'t apply out-of-place transformation to objects which can\'t be copied',
    raises=NotCopyableError,
    strict=True,
)
def test_rigidly_transformed_fails_when_non_copyable(
    points_non_copyable : Union[Points, PointsNonCopyable],
    transform : RigidTransform,
):
    '''Test that rigid transformation are correctly applied out-of-place'''
    _ = points_non_copyable.rigidly_transformed(transform) # no asserts needed, since this line should fail

def test_reset_transformed(
    points : Union[Points, PointsNonCopyable],
    transform : RigidTransform,
):
    '''Test that resetting a rigid transformation in-place return the object to its original state'''
    new_points = points.rigidly_transformed(transform)
    resetted_points = new_points.reset_transformed()

    np.testing.assert_allclose(
        points.positions,
        resetted_points.positions,
        rtol=1E-7,
        atol=1E-10, # DEV: need abs tolerance to be non-zero, since some array values are exactly 0 (fails comparison on MacOS CI)
        strict=True,
    )
    
@pytest.mark.xfail(
    reason='Can\'t apply out-of-place transformation to objects which can\'t be copied',
    raises=NotCopyableError,
    strict=True,
)
def test_reset_transformed_fails_when_non_copyable(
        points_non_copyable : Union[Points, PointsNonCopyable],
    ):
    '''Test that rigid transformation are correctly applied out-of-place'''
    _ = points_non_copyable.reset_transformed() # no asserts needed, since this line should fail

