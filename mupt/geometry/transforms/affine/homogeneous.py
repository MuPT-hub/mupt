'''For handling array conversions to and from homogeneous coordinates'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import numpy as np
from ...arraytypes import Shape, N, M, Dims, DimsPlus, Numeric


def to_homogeneous_coords(positions : np.ndarray[Shape[N, Dims], Numeric], projection : float=1.0) -> np.ndarray[Shape[N, DimsPlus], Numeric]:
    '''Convert an array of N points in D dimensions to one of N points in
    [D + 1] dimensions, with arbitrary uniform projection (1.0 by default)'''
    pad_widths = np.zeros((positions.ndim, 2), dtype=int) # pad nowhere...
    pad_widths[-1, -1] = 1 # ...EXCEPT exactly one layer AFTER the final dimension (and 0 before)
    
    return np.pad(positions, pad_width=pad_widths, mode='constant', constant_values=projection)
      
def from_homogeneous_coords(positions : np.ndarray[Shape[N, DimsPlus], Numeric]) -> np.ndarray[Shape[N, Dims], Numeric]:
    '''Project down from an array of N points in [D + 1] dimensions to an array
    of N points in D dimensions, normalizing out by the homogeneous coordinate'''
    return positions[..., :-1] / positions[..., -1, None] # strip off and normalize by the projective part (via broadcast)

