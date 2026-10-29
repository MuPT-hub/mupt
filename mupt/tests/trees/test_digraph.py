"""Unit tests for anytree-to-networkx interconversion"""

import pytest

from anytree.node import Node
from anytree.iterators import PreOrderIter
from anytree.exporter import DictExporter

from networkx.classes import DiGraph

from mupt.trees.digraph import anytree_to_networkx


RESERVED_ATTR_NAME: str = "value"


def example_tree_simple():
    """
    A small tree with names and custom attributes

    root  (value=0)
    ├── group_1  (value=1)
    ├── group_2  (value=2)
    │   └── atom_0  (value=3)
    └── atom_1  (value=3)
    """
    root = Node("root", **{RESERVED_ATTR_NAME: 0})
    _group_1 = Node("group_1", parent=root, **{RESERVED_ATTR_NAME: 1})
    group_2 = Node("group_2", parent=root, **{RESERVED_ATTR_NAME: 2})
    _atom_0 = Node("atom_0", parent=group_2, **{RESERVED_ATTR_NAME: 3})
    _atom_1 = Node("atom_1", parent=root, **{RESERVED_ATTR_NAME: 4})

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


def example_trees() -> tuple[Node, ...]:
    """Combined set of tree examples useful for parameterize"""
    return (
        example_tree_simple(),
        example_tree_deep(5),
        example_tree_deep(15),
        example_tree_broad(4),
        example_tree_broad(14),
    )


@pytest.mark.parametrize("root", example_trees())
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


def test_trivial_graph() -> None:
    """
    Test that the degenerate case of a childless root
    is correctly converted to the trivial one-node graph
    """
    root = Node("root")
    digraph = anytree_to_networkx(root)

    assert set(digraph.nodes) == {root}
    assert digraph.number_of_edges() == 0


class TestDigraphAttrbutes:
    """
    Bundled tests to ensure Node attributes are transferred
    into their networkx node counterparts on conversion
    """

    def test_node_name(self) -> None:
        """Test that name gets transferred over"""
        root = example_tree_simple()
        digraph = anytree_to_networkx(root)

        for node in PreOrderIter(root):
            assert digraph.nodes[node].get("name") == node.name

    def test_node_custom_attr(self) -> None:
        """Test that custom fields are also copied over"""
        root = example_tree_simple()
        digraph = anytree_to_networkx(root)

        for node in PreOrderIter(root):
            custom_attr_expected = getattr(node, RESERVED_ATTR_NAME)
            custom_attr_actual = digraph.nodes[node].get(RESERVED_ATTR_NAME)
            assert custom_attr_actual == custom_attr_expected

    def test_node_converter(self) -> None:
        """
        Test that conversion of Nodes to other Hashable types in
        DiGraph acts like an isomorphic relabelling
        """
        root = example_tree_simple()
        node_remap = {nd: i for i, nd in enumerate(PreOrderIter(root))}
        digraph = anytree_to_networkx(
            root,
            node_converter=node_remap,
        )

        assert set(digraph.nodes) == set(node_remap.values())

    def test_node_converter_mapping_for_subtree(self) -> None:
        """Only look up exported subtree nodes in the converter mapping"""
        # https://github.com/MuPT-hub/mupt/pull/135#discussion_r4393946172
        root = Node("root")
        branch = Node("branch", parent=root)
        leaf = Node("leaf", parent=branch)
        node_remap = {branch: "branch", leaf: "leaf"}

        digraph = anytree_to_networkx(branch, node_converter=node_remap)

        assert set(digraph.nodes) == {"branch", "leaf"}
        assert set(digraph.edges) == {("branch", "leaf")}

    def test_node_converter_mapping_omits_filtered_parent(self) -> None:
        """Allow filtered parents to be absent from the converter mapping"""
        # https://github.com/MuPT-hub/mupt/pull/135#discussion_r4393946172
        root = Node("root")
        branch = Node("branch", parent=root)
        leaf = Node("leaf", parent=branch)
        node_remap = {root: "root", leaf: "leaf"}

        digraph = anytree_to_networkx(
            root,
            node_converter=node_remap,
            filter_predicate=lambda node: node is not branch,
        )

        assert set(digraph.nodes) == {"root", "leaf"}
        assert digraph.number_of_edges() == 0


class TestDigraphFilter:
    """
    Bundled tests checking that filtering nodes by predicate,
    prunes the expected nodes from the returned tree
    """

    def test_filter_predicate_nodes(self) -> None:
        """Test that filter only includes the desired nodes"""
        root = example_tree_simple()

        def filter_predicate(node) -> bool:
            return not node.name.startswith("group")

        digraph = anytree_to_networkx(root, filter_predicate=filter_predicate)
        assert set(digraph.nodes) == set(filter(filter_predicate, PreOrderIter(root)))

    def test_filter_predicate_edges(self) -> None:
        """
        Test that no edges are placed between nodes whose
        intermediate relative has been filtered out
        """
        root = example_tree_simple()
        nodes_by_name = {nd.name: nd for nd in PreOrderIter(root)}

        def filter_predicate(node) -> bool:
            return not node.name.startswith("group")

        digraph = anytree_to_networkx(root, filter_predicate=filter_predicate)
        assert digraph.has_edge(
            nodes_by_name["root"], nodes_by_name["atom_1"]
        ) and not digraph.has_edge(nodes_by_name["root"], nodes_by_name["atom_0"])


class TestDigraphDepth:
    """Bundled tests to ensure max_depth criterion is respected on traversal"""

    @pytest.mark.parametrize("root", example_trees())
    def test_max_depth_1_root_only(self, root: Node) -> None:
        """Test that max_depth=1 traverses no farther than the root"""
        digraph = anytree_to_networkx(root, max_depth=1)
        assert set(digraph.nodes) == {root}

    @pytest.mark.parametrize("root", example_trees())
    def test_max_depth_2_captures_root_and_children(self, root) -> None:
        """Test that max_depth=2 traverses no farther than root and children"""
        nodes_expected = set.union({root}, set(root.children))
        digraph = anytree_to_networkx(root, max_depth=2)

        assert set(digraph.nodes) == nodes_expected

    @pytest.mark.parametrize("width", (1, 3, 10))
    def test_max_depth_2_shallow_tree(self, width: int) -> None:
        """Test that flat tree with only children is fully exported at max_depth=2"""
        root = example_tree_broad(width)
        digraph = anytree_to_networkx(root, max_depth=2)

        assert digraph.number_of_nodes() == 1 + width

    @pytest.mark.parametrize("depth", (1, 3, 10))
    def test_max_depth_excess(self, depth: int) -> None:
        """
        Test that max_depth no less than of the deepest
        node in the graph does not exclude any nodes
        """
        root = example_tree_deep(depth)
        digraph = anytree_to_networkx(root, max_depth=depth + 1)

        assert digraph.number_of_nodes() == root.size

    def test_custom_node_exporter_max_depth_clamp(self) -> None:
        """
        Test that non-recursive node export (i.e. maxlevel=1) is enforce
        if bespoke DictExporter is passed in to digraph conversion
        """
        root = example_tree_simple()
        exporter = DictExporter(maxlevel=5)

        with pytest.raises(
            ValueError,
            match="Externally-supplied DictExporter must have maxlevel=1",
        ):
            _ = anytree_to_networkx(root, node_exporter=exporter)
