"""Unit tests for trees.subselect"""

import pytest
from typing import Iterable, Optional

from anytree.node import Node

from mupt.trees.subselect import primoprogenitors, NodePredicate


ROOT_KW: str = "base"


@pytest.fixture(scope="function")
def example_tree_with_root() -> tuple[dict[str, Node], Node]:
    """
    A dummy tree for testing subselection routines

    Returns the root of the tree and a dict to each
    node in the tree, each node keyed by its name
    """
    node_name_map: dict[str, str | None] = {
        ROOT_KW: None,
        "tail_0": "base",
        "tail_1": "base",
        "group_0": "tail_0",
        "group_1": "base",
        "moiety_0": "group_0",
        "moiety_1": "group_1",
        "moiety_2": "group_1",
        "atom_0": "group_1",
        "atom_1": "tail_0",
        "atom_2": "moiety_1",
        "atom_3": "base",
    }
    # DEV: slightly brittle; no check to ensure value is already in dict
    node_map: dict[str, Node] = {}
    for node_name, node_name_parent in node_name_map.items():
        node = Node(name=node_name)
        if node_name_parent is not None:
            node.parent = node_map[node_name_parent]
        node_map[node_name] = node

    return node_map, node_map[ROOT_KW]


@pytest.mark.parametrize(
    "predicate,maxlevel,selected_expected_names",
    [
        # Name-based predicates
        (
            lambda node: ("t" in node.name) and (node.name != ROOT_KW),
            None,
            ("tail_0", "tail_1", "moiety_1", "moiety_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: ("t" in node.name) and (node.name != ROOT_KW),
            1,
            ("tail_0", "tail_1", "atom_3"),
        ),
        (
            lambda node: ("t" in node.name) and (node.name != ROOT_KW),
            3,  # entire tree is max depth 3, so doesn't affect result
            ("tail_0", "tail_1", "moiety_1", "moiety_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: node.name == ROOT_KW,
            None,
            (ROOT_KW,),
        ),
        (
            lambda node: node.name == ROOT_KW,
            0,  # root is always traversed, unaffected
            (ROOT_KW,),
        ),
        (
            lambda node: len(node.name) == 6,
            None,
            ("tail_0", "tail_1", "atom_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: len(node.name) == 6,
            1,
            ("tail_0", "tail_1", "atom_3"),
        ),
        (
            lambda node: node.name.endswith("1"),
            None,
            ("atom_1", "tail_1", "group_1"),
        ),
        # Tree hierarchy predicates
        (
            lambda node: not node.children,
            None,
            ("moiety_0", "atom_1", "tail_1", "atom_2", "moiety_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: not node.children,
            2,
            ("atom_1", "tail_1", "moiety_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: len(node.children) == 3,
            None,
            ("group_1",),
        ),
    ],
)
def test_primoprogenitors_correctness(
    example_tree_with_root: tuple[dict[str, Node], Node],
    predicate: NodePredicate[Node],
    maxlevel: Optional[int],
    selected_expected_names: Iterable[str],
) -> None:
    """Test that given tree-predicate pairs return the expected selection"""
    node_map, root = example_tree_with_root
    selected_expected = set(node_map[name] for name in selected_expected_names)
    selected_actual = set(primoprogenitors(root, predicate, maxlevel=maxlevel))

    assert selected_actual == selected_expected


@pytest.mark.parametrize(
    "predicate",
    [
        # TB: same predicates as correctness test, due to dev lazyness :P
        lambda node: ("t" in node.name) and (node.name != ROOT_KW),
        lambda node: node.name == ROOT_KW,
        lambda node: len(node.name) == 6,
        lambda node: node.name.endswith("1"),
        lambda node: not node.children,
        lambda node: len(node.children) == 3,
    ],
)
def test_primoprogenitors_invariant_enforced(
    example_tree_with_root: tuple[dict[str, Node], Node],
    predicate: NodePredicate[Node],
) -> None:
    """
    Test the primoprogenitors enforce the defining invariant that
    no ancestor of a selected node satisfies the selection prediction
    """
    _, root = example_tree_with_root
    for node_selected in primoprogenitors(root, predicate):
        assert not any(predicate(anc) for anc in node_selected.ancestors)
