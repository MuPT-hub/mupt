"""Unit tests for graph visualization helpers and drawing functionality"""

import pytest

from networkx.classes import MultiGraph

from mupt.graphs.visualization import (
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


@pytest.mark.parametrize("base_arc_radius", (0.05, 0.1, 0.25, 1.0))
def test_determine_arc_radii(base_arc_radius: float) -> None:
    """Test that parallel graph arc radii are evenly-spaced and scaled appropriately"""
    EDGE_MULTIPLICITIES: dict[tuple[int, int], int] = {
        (0, 1): 1,  # single edge
        (1, 2): 2,  # double edge
        (0, 2): 3,  # triple edge
        (2, 3): 4,  # quadruple edge
    }
    multigraph = MultiGraph([
        edge
        for edge, multiplicity in EDGE_MULTIPLICITIES.items()
        for _ in range(multiplicity)
    ])

    arc_radii_expected: dict[tuple[int, int, int], float] = {
        (0, 1, 0): 0.0 * base_arc_radius,
        (1, 2, 0): -1 / 2 * base_arc_radius,
        (1, 2, 1): 1 / 2 * base_arc_radius,
        (0, 2, 0): -1.0 * base_arc_radius,
        (0, 2, 1): 0.0 * base_arc_radius,
        (0, 2, 2): 1.0 * base_arc_radius,
        (2, 3, 0): -3 / 2 * base_arc_radius,
        (2, 3, 1): -1 / 2 * base_arc_radius,
        (2, 3, 2): 1 / 2 * base_arc_radius,
        (2, 3, 3): 3 / 2 * base_arc_radius,
    }
    arc_radii_actual = determine_arc_radii(multigraph, base_arc_radius=base_arc_radius)

    assert arc_radii_actual == arc_radii_expected
