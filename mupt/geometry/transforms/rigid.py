'''Utilities for representing and applying rigid transformations to points in 3D space (i.e. working with the Euclidean isometry group E(3))'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform