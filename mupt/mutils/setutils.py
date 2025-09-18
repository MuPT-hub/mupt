'''
Utilities related to generic set-theoretic operations,
including products, relations, and mappings between sets
'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Mapping
from itertools import product as cartesian


def check_bijection() -> None:
    '''Check that two collections of objects have been put into 1-to-1 correspondence with one another'''
    ...