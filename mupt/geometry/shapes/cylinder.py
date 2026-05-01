'''For representing cylindrical, rodlike bodies'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from .shapes import BoundedTransformableShape


class Cylinder(BoundedTransformableShape):
    '''A cylindrical body with arbitrary radius, height, and center'''
    ... # DEV: fill this in
    
Rod = Cylinder # alias for convenience