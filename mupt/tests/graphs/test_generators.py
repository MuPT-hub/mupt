"""Testing that graph generators produce the networks they claim to"""

import pytest

from typing import Optional
from collections import defaultdict

from networkx.classes import Graph, get_node_attributes
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


@pytest.mark.parametrize(
    "graph,label_attr_name,labels_expected",
    [
        # test defaults
        (dendrimer_graph(), "label", {"core", "branch", "leaf"}),
        (
            dendrimer_graph(coord_number=3, num_generations=2),
            "label",
            {"core", "branch", "leaf"},
        ),
        (
            dendrimer_graph(coord_number=3, coord_number_core=2, num_generations=2),
            "label",
            {"core", "branch", "leaf"},
        ),
        # test missing branch labels in degenerate graphs
        (
            dendrimer_graph(coord_number=3, num_generations=1),
            "label",
            {"core", "leaf"},
        ),
        (
            dendrimer_graph(coord_number=1, coord_number_core=2, num_generations=2),
            "label",
            {"core", "leaf"},
        ),
        # test custom label names
        (dendrimer_graph(label_attr_name="other"), "other", {"core", "branch", "leaf"}),
        (
            dendrimer_graph(
                label_attr_name="another",
                core="foo",
                shell_inner="bar",
                shell_outer="baz",
            ),
            "another",
            {"foo", "bar", "baz"},
        ),
        (
            dendrimer_graph(
                label_attr_name="yet another",
                core="foo",
                shell_inner="bar",
                shell_outer="bar",
            ),
            "yet another",
            {"foo", "bar"},
        ),
    ],
)
def test_dendrimer_graph_labels(
    graph: Graph,
    label_attr_name: str,
    labels_expected: set[str],
) -> None:
    """
    Test that the 3 nodes types in a dendrimer graph
    (core, branch, and leaf) are labelled as expected
    """
    print(get_node_attributes(graph, label_attr_name))
    labels_actual: set[str] = set(get_node_attributes(graph, label_attr_name).values())
    assert labels_actual == labels_expected


@pytest.mark.parametrize(
    "coord_number,coord_number_core,num_generations",
    [
        (2, 3, 4),
        (2, None, 4),
        (3, 5, 2),
        (3, None, 2),
        (1, 2, 5),
        (1, None, 5),
        (5, 3, 1),
        (5, None, 1),
        (5, 3, 2),
        (5, None, 2),
        (1, 3, 2),
        (1, None, 2),
    ],
)
def test_dendrimer_graph_degrees(
    coord_number: int,
    coord_number_core: Optional[int],
    num_generations: int,
) -> None:
    """
    Test that the degree of every node type matches what one
    would expected from the specified coordination numbers
    """
    LABEL_ATTR: str = "label"  # creative, I know
    CORE_NAME: str = "core"
    BRANCH_NAME: str = "branch"
    LEAF_NAME: str = "leaf"

    dendr = dendrimer_graph(
        coord_number=coord_number,
        coord_number_core=coord_number_core,
        num_generations=num_generations,
        label_attr_name=LABEL_ATTR,
        core=CORE_NAME,
        shell_inner=BRANCH_NAME,
        shell_outer=LEAF_NAME,
    )

    degrees_by_node_type = defaultdict(set)
    for node, label in get_node_attributes(dendr, LABEL_ATTR).items():
        degrees_by_node_type[label].add(dendr.degree[node])
    degrees_by_node_type = dict(degrees_by_node_type)

    if coord_number_core is None:
        coord_number_core = coord_number

    degrees_by_node_type_expected = {
        CORE_NAME: {coord_number_core},
        LEAF_NAME: {1},
    }
    if (num_generations > 1) and (coord_number > 1):
        # branch nodes may not be present for certain degenerate dendrimer graphs
        degrees_by_node_type_expected[BRANCH_NAME] = {coord_number}

    print(degrees_by_node_type_expected, degrees_by_node_type)
    assert degrees_by_node_type == degrees_by_node_type_expected
