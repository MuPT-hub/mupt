"""Unit tests for anytree-to-networkx interconversion"""

import pytest

from anytree.node import Node
from anytree.iterators import PreOrderIter
from networkx.classes import DiGraph

from mupt.trees.digraph import anytree_to_networkx


def test_anytree_to_networkx() -> None:
    """Test anytree Node trees are correctly converted to networkx directed graphs"""
    ...


def examples_tree_simple():
    """
    A small tree with names and custom attributes

    root
    ├── group_1  (value=1)
    └── group_2  (value=2)
        └── atom_0  (value=3)
    """
    root = Node("root")
    _group_1 = Node("group_1", parent=root, value=1)
    group_2 = Node("group_2", parent=root, value=2)
    _atom_0 = Node("atom_0", parent=group_2, value=3)

    return root


def example_tree_broad(width: int) -> Node:
    """A flat tree with arbitrarily many children below the root"""
    root = Node(name="root")
    for i in range(width):
        _ = Node(name=f"child_{i}", parent=root)

    return root


def example_tree_deep(depth: int) -> Node:
    """A single-branch tree of arbitrary depth"""
    root = Node(name="root")

    node_prev = root
    for i in range(depth):
        node_prev = Node(name=f"anc_depth={i}", parent=node_prev)

    return root


@pytest.mark.parametrize(
    "root",
    [
        examples_tree_simple(),
        example_tree_deep(5),
        example_tree_deep(15),
        example_tree_broad(4),
        example_tree_broad(14),
    ],
)
class TestDiGraphStructure:
    """
    Bundled tests which ensure connectivity, return types, and node counts
    are accurate at a baseline, i.e. without any filtering or attribute info
    """

    def test_return_type(self, root: Node) -> None:
        """Test that all conversion return DiGraphs"""
        digraph = anytree_to_networkx(root)
        assert isinstance(digraph, DiGraph)

    def test_node_content(self, root: Node) -> None:
        """Test that, without any filtering, all nodes are coverted over"""
        digraph = anytree_to_networkx(root)

        # size check not strictly necessary; if two sets are identically
        # equal, then they must have the same cardinality a fortiori
        assert root.size == digraph.number_of_nodes()
        assert set(digraph.nodes) == set(PreOrderIter(root))
        ...

    def test_edge_content(self, root: Node) -> None:
        """
        Test that, without any filtering, all directed edges are converted over
        and point in the correct direction (i.e. always from parent -> child)
        """
        digraph = anytree_to_networkx(root)

        edges_expected: set[tuple[Node, Node]] = {
            (node.parent, node)
            for node in PreOrderIter(root)
            if node.parent is not None
        }
        assert set(digraph.edges()) == edges_expected


class TestDigraphFilter:
    """
    Bundled tests checking that filtering nodes by attributes, predicate, or depth
    prunes the expected nodes from the returned tree
    """

    ...
