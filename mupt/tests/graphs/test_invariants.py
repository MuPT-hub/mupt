"""Unit tests for graph invariants"""

import pytest
from dataclasses import dataclass

from networkx.classes import Graph, DiGraph
from networkx.algorithms import disjoint_union
from networkx.generators import (
    null_graph,
    empty_graph,
    trivial_graph,
    cycle_graph,
    path_graph,
    star_graph,
    tadpole_graph,
    heawood_graph,  # TB: can be any branched graph for tests; chose a personal favorite
    caveman_graph,
)

from mupt.graphs.graphtypes import Node
from mupt.graphs.invariants import (
    is_empty,
    is_indiscrete,
    is_unbranched,
    is_branched,
    is_linear,
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
    is_linear: bool


def graph_invariant_examples() -> tuple[GraphInvariantExample, ...]:
    """Examples which are to be reused in tests here"""
    return (
        # empty graph (no nodes or edges)
        GraphInvariantExample(
            null_graph(),
            is_indiscrete=True,
            is_empty=True,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        GraphInvariantExample(
            null_graph(create_using=DiGraph),
            is_indiscrete=True,
            is_empty=True,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        GraphInvariantExample(
            Graph(),
            is_indiscrete=True,
            is_empty=True,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        # trivial graph (one node, no edges)
        GraphInvariantExample(
            trivial_graph(),
            is_indiscrete=True,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        # alternate form of the empty graph
        GraphInvariantExample(
            path_graph(0),
            is_indiscrete=True,
            is_empty=True,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        # alternate form of the trivial graph
        GraphInvariantExample(
            path_graph(1),
            is_indiscrete=True,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        GraphInvariantExample(
            path_graph(3),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        GraphInvariantExample(
            path_graph(3, create_using=DiGraph),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        # test with disconnected components (in this case, isolated nodes)
        GraphInvariantExample(
            empty_graph(4),
            is_indiscrete=True,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        # Cyclic graphs - unbranched, but not linear
        GraphInvariantExample(
            cycle_graph(5),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        GraphInvariantExample(
            cycle_graph(5, create_using=DiGraph),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        # graphs with loops
        ## 2-cycle "head" is still linear
        GraphInvariantExample(
            tadpole_graph(2, 3),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        ## 3-cycle "head" introduces branching
        GraphInvariantExample(
            tadpole_graph(3, 3),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=False,
            is_linear=False,
            is_branched=True,
        ),
        # graphs with branches
        ## alternate form of the trivial graph
        GraphInvariantExample(
            star_graph(0),
            is_indiscrete=True,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        ## equivalent to a 2-node path graph
        GraphInvariantExample(
            star_graph(1),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        ## equivalent to a 3-node path graph
        GraphInvariantExample(
            star_graph(2),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=True,
            is_linear=True,
            is_branched=False,
        ),
        ## branching of star graph finally "kicks in"
        GraphInvariantExample(
            star_graph(3),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=False,
            is_linear=False,
            is_branched=True,
        ),
        GraphInvariantExample(
            star_graph(3, create_using=DiGraph),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=False,
            is_linear=False,
            is_branched=True,
        ),
        GraphInvariantExample(
            heawood_graph(),
            is_indiscrete=False,
            is_empty=False,
            is_unbranched=False,
            is_linear=False,
            is_branched=True,
        ),
    )


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
def test_is_linear(graph_example: GraphInvariantExample) -> None:
    """Test that the is_linear() invariant is calculated correctly"""
    assert is_linear(graph_example.graph) == graph_example.is_linear


@pytest.mark.parametrize("graph_example", graph_invariant_examples())
def test_is_branched(graph_example: GraphInvariantExample) -> None:
    """Test that the is_branched() invariant is calculated correctly"""
    assert is_branched(graph_example.graph) == graph_example.is_branched


# Subgraph enumeration
@pytest.mark.parametrize(
    "graph,component_nodes_expected",
    [
        (path_graph("abcd"), ({"a", "b", "c", "d"},)),
        (caveman_graph(3, 3), ({0, 1, 2}, {3, 4, 5}, {6, 7, 8})),
        (empty_graph([0.5772, 2.7182, 3.1415]), ({0.5772}, {2.7182}, {3.1415})),
        (
            # TB: wanted to test this with directed cycles, only to find
            # networkx does not implement connected_components for DiGraphs :(
            disjoint_union(cycle_graph(3), cycle_graph(5)),
            ({0, 1, 2}, {3, 4, 5, 6, 7}),
        ),
    ],
)
def test_components(
    graph: Graph,
    component_nodes_expected: tuple[set[Node]],
) -> None:
    """
    Test that connected components of a graph are correctly
    identified AND returned as same type as that graph
    """
    for component in components(graph):
        assert type(component) is type(graph)
        assert set(component) in component_nodes_expected


@pytest.mark.parametrize(
    "graph,termini_expected",
    [],
)
def test_termini(graph: Graph, termini_expected: set[Node]) -> None:
    """Test that identification of terminal/leaf nodes returns the expected results"""
    termini_actual: set[Node] = set(termini(graph))
    assert termini_actual == termini_expected
