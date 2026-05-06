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


def test_volume(shape : BoundedShape) -> None:
    ...

def test_scaling(shape : BoundedShape) -> None:
    ...

def test_containment(shape : BoundedShape) -> None:
    ...

def test_rigid_transforms(shape : BoundedTransformableShape) -> None:
    ...