'''For representing coordinate arrays and transformations'''

from typing import Generic, TypeVar
from numbers import Number, Real

Numeric = TypeVar('Numeric', bound=Number)

import numpy as np
from scipy.spatial.transform import Rotation

 
class Coordinates(Generic[Numeric]):
    '''Encapsulation class for storing sets of coordinates and performing transfomations on those coordinates'''
    pass

class Conformer(Coordinates[Real]):
    '''For representing the positions of units of a molecules'''
    pass

    # incorporate binding to atoms/masses 