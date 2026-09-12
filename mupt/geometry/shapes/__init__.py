"""For representing spatial information about bounded and rigid bodies"""

from .shapes import (
    BoundedShape as BoundedShape,
    BoundedTransformableShape as BoundedTransformableShape,
    Shaped as Shaped,
)
from .pointcloud import PointCloud as PointCloud
from .ellipsoid import Sphere as Sphere, Ellipsoid as Ellipsoid
from .cylinder import Rod as Rod, Cylinder as Cylinder
from .visualize import visualize_shape as visualize_shape
