'''For applying coordinates to molecular topologies'''

from numbers import Real
from ..geometry.coordinates import Coordinates


class Conformer(Coordinates[Real]):
    '''For representing the positions of units of a molecules'''
    pass

    # incorporate binding to atoms/masses 