"""Unit tests for visualization of Primitives"""

import pytest

from typing import Generator

from matplotlib.axes import Axes
from matplotlib.pyplot import subplots, close as close_figure

from mupt.mupr.primitives.visualize import visualize_cross_section, visualize_hierarchy
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.predicates import is_subdominant
from mupt.mupr.connection.connectors import Connector, AttachmentPoint


@pytest.fixture
def hierarchy_axes() -> Generator[Axes, None, None]:
    """Provide drawing axes and clean up the figure even when drawing fails"""
    fig, ax = subplots()
    try:
        yield ax
    finally:
        close_figure(fig)


def test_visualize_cross_section(hierarchy_axes: Axes) -> None:
    """Smoke test to ensure cross-section graph render works"""
    conn_0 = Connector(AttachmentPoint({0}), AttachmentPoint({1}))
    conn_0c = conn_0.counterpart()

    conn_1 = conn_0c.copy()
    conn_1c = conn_1.counterpart()

    simple_0 = SimplePrimitive(label="simple_0", connections=(conn_0,))
    simple_1 = SimplePrimitive(label="simple_1", connections=(conn_0c, conn_1c))
    simple_2 = SimplePrimitive(label="simple_2", connections=(conn_1,))
    comp = CompositePrimitive(label="comp", children=[simple_2])
    root = RootPrimitive(label="root", children=[simple_0, simple_1, comp])

    simple_1.connect_neighbor(simple_0, our_connector=conn_0c, their_connector=conn_0)
    simple_1.connect_neighbor(simple_2, our_connector=conn_1c, their_connector=conn_1)

    xsec = root.cross_section(is_subdominant)
    assert xsec.number_of_nodes() == 3
    _ax = visualize_cross_section(
        root,
        cross_section=xsec,
        use_shapes_for_pos=True,
        ax=hierarchy_axes,
    )


def test_visualize_hierarchy(hierarchy_axes: Axes) -> None:
    """Smoke test to ensure hierarchy tree render works"""
    simple_0 = SimplePrimitive()
    simple_1 = SimplePrimitive()
    root = RootPrimitive(label="root", children=[simple_0, simple_1])

    _ax = visualize_hierarchy(root, ax=hierarchy_axes)
