'''Definitions of positions in particular coordinate system and bases'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from .reference import origin
from .basis import (
    are_linearly_independent,
    is_diagonal,
    is_rowspace_mutually_orthogonal,
    is_columnspace_mutually_orthogonal,
    is_orthogonal,
)
from .directions import (
    random_vector,
    random_unit_vector,
    random_orthogonal_vector,
)
from .local import compute_local_coordinates