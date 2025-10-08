'''Indicators for coordinate axis-specific operations and indexing'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from enum import Enum

class CoordAxis(Enum):
    '''
    For making clear when a particular coordinate direction 
    is chosen for a task, particularly when indexing
    '''
    X = 0
    Y = 1
    Z = 2
    W = 3
    
    # lowercase aliases for the lazy
    x = 0
    y = 1
    z = 2
    w = 3
