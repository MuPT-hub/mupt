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
    "predicate,selected_expected_names",
    [
        # Name-based predicates
        (
            lambda node: ("t" in node.name) and (node.name != ROOT_KW),
            ("tail_0", "tail_1", "moiety_1", "moiety_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: node.name == ROOT_KW,
            (ROOT_KW,),
        ),
        (
            lambda node: len(node.name) == 6,
            ("tail_0", "tail_1", "atom_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: node.name.endswith("1"),
            ("atom_1", "tail_1", "group_1"),
        ),
        # Tree hierarchy predicates
        (
            lambda node: not node.children,
            ("moiety_0", "atom_1", "tail_1", "atom_2", "moiety_2", "atom_0", "atom_3"),
        ),
        (
            lambda node: len(node.children) == 3,
            ("group_1",),
        ),
    ],
)
def test_primoprogenitors_correctness(
    example_tree_with_root: tuple[dict[str, Node], Node],
    predicate: NodePredicate,
    selected_expected_names: Iterable[str],
    # TODO: add tests for maxlevel
    maxlevel: Optional[int] = None,
) -> None:
    """Test that given tree-predicate pairs return the expected selection"""
    node_map, root = example_tree_with_root
    selected_expected = set(node_map[name] for name in selected_expected_names)
    selected_actual = set(primoprogenitors(root, predicate, maxlevel=maxlevel))

    assert selected_actual == selected_expected
