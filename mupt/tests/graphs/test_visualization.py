"""Unit tests for graph visualization helpers and drawing functionality"""

import pytest
import logging

from networkx.classes import DiGraph, MultiGraph
from networkx.generators import star_graph
from matplotlib.pyplot import subplots

from mupt.graphs.visualization import (
    default_graph_axes,
    determine_arc_radii,
    determine_arc_styles,
    draw_networkx_with_arcs,
    draw_networkx_tree,
)


@pytest.fixture()
def epitomic_multigraph() -> MultiGraph:
    """A multigraph with single, double, triple, and quadruple parallel edges"""
    edge_multiplicities: dict[tuple[int, int], int] = {
        (0, 1): 1,  # single edge
        (1, 2): 2,  # double edge
        (0, 2): 3,  # triple edge
        (2, 3): 4,  # quadruple edge
    }
    return MultiGraph([
        edge
        for edge, multiplicity in edge_multiplicities.items()
        for _ in range(multiplicity)
    ])


@pytest.fixture
def example_tree() -> DiGraph:
    """A directed graph representing a simple tree"""
    return star_graph(5, create_using=DiGraph)


@pytest.mark.parametrize("margins", (0.05, 0.1, 0.25))
def test_default_graph_axes(margins: float) -> None:
    """Test axes are produced and have the desired margins"""
    ax = default_graph_axes(margins=margins)

    assert ax.margins() == (margins, margins)
    BOUNDS_EXPECTED: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    for bound_expected, bound_actual in zip(BOUNDS_EXPECTED, ax.get_position().bounds):
        assert bound_actual == bound_expected


@pytest.mark.parametrize("base_arc_radius", (0.05, 0.1, 0.25, 1.0))
def test_determine_arc_radii(
    epitomic_multigraph: MultiGraph,
    base_arc_radius: float,
) -> None:
    """Test that parallel graph arc radii are evenly-spaced and scaled appropriately"""
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
    arc_radii_actual = determine_arc_radii(
        epitomic_multigraph,
        base_arc_radius=base_arc_radius,
    )

    assert arc_radii_actual == arc_radii_expected


def test_determine_arc_styles(epitomic_multigraph: MultiGraph) -> None:
    """Test that parallel graph arc radii strings are correctly interpolated"""
    # DEV: interpolating, rather than hard-coding values into strings,
    # since float rounding error DOES actually impact how the final
    # style string is produced in some cases
    arc_radii_expected: dict[tuple[int, int, int], str] = {
        (0, 1, 0): f"arc3,rad={0.0}",
        (1, 2, 0): f"arc3,rad={-0.5}",
        (1, 2, 1): f"arc3,rad={0.5}",
        (0, 2, 0): f"arc3,rad={-1.0}",
        (0, 2, 1): f"arc3,rad={0.0}",
        (0, 2, 2): f"arc3,rad={1.0}",
        (2, 3, 0): f"arc3,rad={-1.5}",
        (2, 3, 1): f"arc3,rad={-0.5}",
        (2, 3, 2): f"arc3,rad={0.5}",
        (2, 3, 3): f"arc3,rad={1.5}",
    }
    arc_radii_actual = determine_arc_styles(
        epitomic_multigraph,
        base_arc_radius=1.0,  # test as normalized
    )

    assert arc_radii_actual == arc_radii_expected


def test_draw_networkx_with_arcs(epitomic_multigraph: MultiGraph) -> None:
    """Smoke test for graph drawing with differentiated multigraph arcs"""
    _ = draw_networkx_with_arcs(epitomic_multigraph)


def test_draw_networkx_tree(example_tree: DiGraph) -> None:
    """Smoke test for graph drawing with differentiated multigraph arcs"""
    _ = draw_networkx_tree(example_tree)


def test_draw_networkx_with_arcs_on_axis(epitomic_multigraph: MultiGraph) -> None:
    """
    Smoke test for graph drawing with differentiated multigraph arcs
    onto a pre-existing Axes. Ensures graph is actually drawn to that axis
    """
    _fig, ax = subplots()
    assert not ax.has_data()

    # subvert plt.gca() mechanism, which gives false positive test pass
    _fig_decoy, ax_decoy = subplots()

    # Don't re-coolect ax; want to see if the ORIGINAL Axes object is drawn on
    _ = draw_networkx_with_arcs(epitomic_multigraph, ax=ax)
    logging.warning(ax.has_data())
    assert ax.has_data()


def test_draw_networkx_tree_on_axis(example_tree: DiGraph) -> None:
    """
    Smoke test for graph drawing with differentiated multigraph arcs
    onto a pre-existing Axes. Ensures graph is actually drawn to that axis
    """
    _fig, ax = subplots()
    assert not ax.has_data()

    # subvert plt.gca() mechanism, which gives false positive test pass
    _fig_decoy, ax_decoy = subplots()

    # Don't re-coolect ax; want to see if the ORIGINAL Axes object is drawn on
    _ = draw_networkx_tree(example_tree, ax=ax)
    logging.error(ax.has_data())
    assert ax.has_data()
