"""Unit tests for graph visualization helpers and drawing functionality"""

import pytest  # noqa: F401

from mupt.graphs.visualization import (  # noqa: F401
    default_graph_axes,
    determine_arc_radii,
)


@pytest.mark.parametrize("margins", (0.05, 0.1, 0.25))
def test_default_graph_axes(margins: float) -> None:
    """Test axes are produced and have the desired margins"""
    ax = default_graph_axes(margins=margins)

    assert ax.margins() == (margins, margins)
    BOUNDS_EXPECTED: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    for bound_expected, bound_actual in zip(BOUNDS_EXPECTED, ax.get_position().bounds):
        assert bound_actual == bound_expected
