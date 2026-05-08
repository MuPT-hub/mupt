'''For representing spatial information about bounded and rigid bodies'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from .shapes import BoundedShape, BoundedTransformableShape, Shaped
from .pointcloud import PointCloud
from .ellipsoid import Sphere, Ellipsoid
from .cylinder import Rod, Cylinder
from .visualize import visualize_shape