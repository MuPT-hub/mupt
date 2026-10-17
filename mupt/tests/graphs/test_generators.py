"""Testing that graph generators produce the networks they claim to"""

import pytest  # noqa: F401

from networkx.classes import Graph
from networkx.algorithms import is_isomorphic

from mupt.graphs.generators import (  # noqa: F401
    path_graph_from_sequence,
    dendrimer_graph,
    noodle_graph,
)


# Reference dendrimer graph networkx
## TB: pretty easy to implement procedural build these with varying parameters,
## but that sort of defeats the point of a test against a known case
## Painstakingly compiled examples manually to provide true oracle for tests
def dendrimer_graph_expected_224() -> Graph:
    """
    Expected dendrimer graph for:
    - 2-functional branches
    - 2-functional core
    - 4 generation depth
    """
    # TB: any 2-2-g combo ends up being isomorphic to a path graph with 2*g+1 nodes
    return Graph([
        ((0, 3), (0, 2)),
        ((0, 1), (0, 2)),
        ((0, 1), (0, 0)),
        (0, (0, 0)),
        (0, (1, 0)),
        ((1, 1), (1, 0)),
        ((1, 2), (1, 1)),
        ((1, 3), (1, 2)),
    ])


def dendrimer_graph_expected_3n2(coord_number_core: int) -> Graph:
    """
    Expected dendrimer graph for:
    - 3-functional branches
    - n-functional core
    - 2 generation depth
    """
    dendr = Graph()
    CORE_IDX: int = 0

    dendr.add_node(CORE_IDX)
    for branch_num in range(coord_number_core):
        dendr.add_edges_from([
            (0, (branch_num, 0)),
            ((branch_num, 0), (branch_num, 1)),
            ((branch_num, 0), (branch_num, 2)),
        ])

    return dendr


def dendrimer_graph_examples_with_references() -> tuple[tuple[Graph, Graph], ...]:
    """Coi=mpiled expected-actual pairs reused in tests requiring reference a Graph"""
    return (
        (
            dendrimer_graph_expected_224(),
            dendrimer_graph(coord_number=2, coord_number_core=2, num_generations=4),
        ),
        (
            dendrimer_graph_expected_3n2(3),
            dendrimer_graph(coord_number=3, num_generations=2),
        ),
        (
            dendrimer_graph_expected_3n2(2),
            dendrimer_graph(coord_number=3, coord_number_core=2, num_generations=2),
        ),
        (
            dendrimer_graph_expected_3n2(5),
            dendrimer_graph(coord_number=3, coord_number_core=5, num_generations=2),
        ),
    )


# Tests proper
@pytest.mark.parametrize(
    "dendr_graph_expected,dendr_graph_actual",
    dendrimer_graph_examples_with_references(),
)
def test_dendrimer_graph_networkx(
    dendr_graph_expected: Graph,
    dendr_graph_actual: Graph,
) -> None:
    """Test that the connectivity of a dendrimer graph is assigned as expected"""
    assert is_isomorphic(dendr_graph_actual, dendr_graph_expected)


@pytest.mark.parametrize(
    "dendr_graph_expected,dendr_graph_actual",
    dendrimer_graph_examples_with_references(),
)
def test_dendrimer_graph_indices(
    dendr_graph_expected: Graph,
    dendr_graph_actual: Graph,
) -> None:
    """Test that the nodes and edges in a dendrimer graph are assigned as expected"""
    assert set(dendr_graph_actual.nodes) == set(dendr_graph_expected.nodes)

    # normalize to ignore order of nodes in each edge (irrelevant for comparison)
    unordered_eges_expected = {frozenset(edge) for edge in dendr_graph_actual.edges}
    unordered_eges_actual = {frozenset(edge) for edge in dendr_graph_actual.edges}

    assert unordered_eges_actual == unordered_eges_expected


def test_dendrimer_graph_labels() -> None:
    """
    Test that the 3 nodes types in a dendrimer graph
    (core, branch, and leaf) are labelled as expected
    """
    ...


def test_dendrimer_graph_degrees() -> None:
    """
    Test that the degree of every node type matches what one
    would expected from the specified coordination numbers
    """
    ...
