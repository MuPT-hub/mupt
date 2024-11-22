'''For representing coordinate arrays and transformations'''

from typing import Generic, TypeVar
from numbers import Number

Numeric = TypeVar('Numeric', bound=Number)

import numpy as np
from scipy.spatial.transform import Rotation

 
class Coordinates(Generic[Numeric]):
    '''Encapsulation class for storing sets of coordinates and performing transfomations on those coordinates'''
    pass
