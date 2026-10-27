"""Unit tests for heading indicators"""

import pytest

from mupt.builders.heading import TraversalDirection


@pytest.mark.parametrize(
    "traversal_direction,complement_expected",
    [
        (TraversalDirection.AMBI, TraversalDirection.AMBI),
        (TraversalDirection.ANTERO, TraversalDirection.RETRO),
        (TraversalDirection.RETRO, TraversalDirection.ANTERO),
    ],
)
def test_traversal_complement(
    traversal_direction: TraversalDirection,
    complement_expected: TraversalDirection,
) -> None:
    """
    Test that TraversalDirection.complement
    returns expected complementary direction
    """
    complement_actual = TraversalDirection.complement(traversal_direction)
    assert complement_actual == complement_expected
