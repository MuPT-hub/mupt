"""Tests for the mupt.mupr.primitives module"""

from typing import Generator

import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from mupt.mupr.primitives import Primitive, visualize_hierarchy


def test_hierarchy_tree_includes_constructor_children() -> None:
    """Keep constructor-supplied children in the exported hierarchy"""
    # https://github.com/MuPT-hub/mupt/pull/135#discussion_r4393946041
    child = Primitive(label="child")
    root = Primitive(label="root", children=[child])

    graph = root.hierarchy_tree()

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge(root, child)


def test_hierarchy_tree_preserves_equivalent_siblings() -> None:
    """Keep separate hierarchy nodes for equivalent sibling Primitives"""
    # https://github.com/MuPT-hub/mupt/pull/135#discussion_r4328033416
    root = Primitive(label="root")
    left = Primitive()
    right = Primitive()
    root.attach_child(left)
    root.attach_child(right)

    graph = root.hierarchy_tree()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


@pytest.fixture
def hierarchy_axes() -> Generator[Axes, None, None]:
    """Provide drawing axes and clean up the figure even when drawing fails"""
    fig, ax = plt.subplots()
    try:
        yield ax
    finally:
        plt.close(fig)


def test_visualize_hierarchy_with_converted_nodes(hierarchy_axes: Axes) -> None:
    """Draw converted node IDs with the default labeling options"""
    # https://github.com/MuPT-hub/mupt/pull/135#discussion_r4344040109
    root = Primitive(label="root")
    graph = root.hierarchy_tree(node_converter=lambda node: node.label)

    ax = visualize_hierarchy(
        root,
        hierarchy_tree=graph,
        pos={"root": (0, 0)},
        ax=hierarchy_axes,
    )

    assert ax is hierarchy_axes
    assert [text.get_text() for text in ax.texts] == ["root"]


@pytest.mark.parametrize(
    "selection", ("depth_limit", "filtered_child", "filtered_root")
)
def test_visualize_hierarchy_labels_only_included_nodes(
    selection: str,
    hierarchy_axes: Axes,
) -> None:
    """Draw partial hierarchies without labels for excluded nodes"""
    # https://github.com/MuPT-hub/mupt/pull/135#discussion_r4393946104
    root = Primitive(label="root")
    child = Primitive(label="child")
    root.attach_child(child)

    if selection == "depth_limit":
        graph = root.hierarchy_tree(max_depth=1)
    elif selection == "filtered_child":
        graph = root.hierarchy_tree(filter_predicate=lambda node: node is root)
    else:
        graph = root.hierarchy_tree(filter_predicate=lambda node: node is child)

    assert graph.number_of_nodes() == 1
    ax = visualize_hierarchy(
        root,
        hierarchy_tree=graph,
        pos={node: (0, 0) for node in graph},
        ax=hierarchy_axes,
    )

    assert ax is hierarchy_axes
    assert len(ax.texts) == graph.number_of_nodes()
