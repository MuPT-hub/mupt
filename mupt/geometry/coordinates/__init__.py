"""Definitions of positions in particular coordinate system and bases"""

from .reference import origin as origin
from .basis import (
    are_linearly_independent as are_linearly_independent,
    is_diagonal as is_diagonal,
    is_rowspace_mutually_orthogonal as is_rowspace_mutually_orthogonal,
    is_columnspace_mutually_orthogonal as is_columnspace_mutually_orthogonal,
    is_orthogonal as is_orthogonal,
)
from .directions import (
    random_vector as random_vector,
    random_unit_vector as random_unit_vector,
    random_orthogonal_vector as random_orthogonal_vector,
)
from .local import compute_local_coordinates as compute_local_coordinates
