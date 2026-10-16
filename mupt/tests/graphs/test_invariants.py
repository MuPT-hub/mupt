"""Unit tests for graph invariants"""

import pytest
from dataclasses import dataclass

from networkx.classes import Graph

from mupt.graphs.graphtypes import Node
from mupt.graphs.invariants import (
    is_empty,
    is_indiscrete,
    is_unbranched,
    is_branched,
    components,
    termini,
)


@dataclass
class GraphInvariantExample:
    """Encapsulation to allow reuse the same graph examples across multiple tests"""

    graph: Graph
    is_indiscrete: bool
    is_empty: bool
    is_unbranched: bool
    is_branched: bool


def graph_invariant_examples() -> tuple[GraphInvariantExample, ...]:
    """Examples which are to be reused in tests here"""
    return ()


# Graph predicates
@pytest.mark.parametrize("graph_example", graph_invariant_examples())
def test_is_indiscrete(graph_example: GraphInvariantExample) -> None:
    """Test that the is_indiscrete() invariant is calculated correctly"""
    assert is_indiscrete(graph_example.graph) == graph_example.is_indiscrete


@pytest.mark.parametrize("graph_example", graph_invariant_examples())
def test_is_empty(graph_example: GraphInvariantExample) -> None:
    """Test that the is_empty() invariant is calculated correctly"""
    assert is_empty(graph_example.graph) == graph_example.is_empty


@pytest.mark.parametrize("graph_example", graph_invariant_examples())
def test_is_unbranched(graph_example: GraphInvariantExample) -> None:
    """Test that the is_unbranched() invariant is calculated correctly"""
    assert is_unbranched(graph_example.graph) == graph_example.is_unbranched


@pytest.mark.parametrize("graph_example", graph_invariant_examples())
def test_is_branched(graph_example: GraphInvariantExample) -> None:
    """Test that the is_branched() invariant is calculated correctly"""
    assert is_branched(graph_example.graph) == graph_example.is_branched


# Subgraph enumeration
@pytest.mark.parametrize(
    "graph,component_nodes_expected",
    [],
)
def test_components(
    graph: Graph,
    component_nodes_expected: frozenset[set[Node]],
) -> None:
    """
    Test that connected components of a graph are correctly
    identified AND returned as same type as that graph
    """
    for component in components(graph):
        assert type(component) is type(graph)
        assert set(component.nodes) in component_nodes_expected


@pytest.mark.parametrize(
    "graph,termini_expected",
    [],
)
def test_termini(graph: Graph, termini_expected: set[Node]) -> None:
    """Test that identification of terminal/leaf nodes returns the expected results"""
    termini_actual: set[Node] = set(termini(graph))
    assert termini_actual == termini_expected
