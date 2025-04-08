'''For applying coordinates to molecular topologies'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from numbers import Real
from ..geometry.coordinates import Coordinates


class Conformer(Coordinates[Real]):
    '''For representing the positions of units of a molecules'''
    pass

    # incorporate binding to atoms/masses 