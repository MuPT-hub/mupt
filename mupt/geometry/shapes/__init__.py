'''For representing spatial information about bounded and rigid bodies'''

from .shapes import BoundedShape, BoundedTransformableShape, Shaped
from .pointcloud import PointCloud
from .ellipsoid import Sphere, Ellipsoid
from .cylinder import Rod, Cylinder
from .visualize import visualize_shape