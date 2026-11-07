"""Unit tests for Connectors, AttachmentPoints, and connection-related utilities"""

from typing import Hashable, Optional
from mupt.mupr.connection.connectors import Connector, AttachmentPoint

import numpy.random as np_random


def randomly_placed_connector(
    anchor_marker: Hashable = 0,
    linker_marker: Hashable = 1,
    L: float = 5.0,
    rng: Optional[np_random.Generator] = None,
) -> Connector:
    """
    Produce a marked Connector with both attachment point
    positions chosen uniformly randomly from [-L, L]**3
    """
    sampler = np_random if rng is None else rng
    return Connector(
        anchor=AttachmentPoint(
            {anchor_marker},
            attachment=anchor_marker,
            position=sampler.uniform(low=-L, high=L, size=3),
        ),
        linker=AttachmentPoint(
            {linker_marker},
            attachment=linker_marker,
            position=sampler.uniform(low=-L, high=L, size=3),
        ),
    )
