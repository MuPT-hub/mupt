"""Unit tests for BoundedShape classes"""

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import pytest 

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from mupt.geometry.shapes import (
    visualize_shape,
    BoundedShape,
    PointCloud,
    Sphere,
    Ellipsoid,
)