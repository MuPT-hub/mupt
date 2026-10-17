"""Routines for creating graphs from simplified non-graph inputs"""

from typing import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Optional,
)

from itertools import count
from functools import reduce

from networkx.classes import Graph, DiGraph
from networkx.generators import path_graph, balanced_tree
from networkx.algorithms import compose, union as graph_union
from networkx.relabel import relabel_nodes

from .graphtypes import GraphLike


def path_graph_from_sequence(
    sequence: Collection[Hashable],
    label_attr: str = "label",
    create_using: type[GraphLike] = Graph,
) -> GraphLike:
    """
    Generate a path graph from a sequence of labels

    Keeps identical labels as distinct node,
    allowing labels to be reused within sequence

    Parameters
    ----------
    sequence : Iterable[Hashable]
        The ordered sequence of labels to assign to nodes
        Labels can be reused as often as one likes without being
    label_attr : str, default 'label'
        The name of the attribute on each node to bind the label value to
    create_using, type[GraphLike], default Graph
        The type of networkx Graph to use to instantiate the path graph

    Returns
    -------
    sequence_graph : GraphLike
        The requisite graph, whose type is that of `create_using`
        Nodes in sequence_graph are the integer index of the label,
        while the labels themselves are bound to `label_attr` on each node
    """
    sequence_graph = path_graph(len(sequence), create_using=create_using)
    for i, label in enumerate(sequence):
        sequence_graph.nodes[i][label_attr] = label

    return sequence_graph


# TODO: handle index uniquification over union of path_graph_from_sequence
# perhaps even absorb path_graphs/noodle_graphs
# TB: looking for disjoint_union, if restricting to int-only indices


def path_graphs(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[Graph] = Graph,
) -> Generator[Graph, None, None]:
    """
    Generate a sequence of path graphs according to a
    provided sequence of lengths and labelling scheme
    """
    if node_labels is None:
        node_labels = count(start=0, step=1)

    for chain_length in chain_lengths:
        yield path_graph(
            (next(node_labels) for _ in range(chain_length)),
            create_using=create_using,
        )


def noodle_graph(
    chain_lengths: Iterable[int],
    node_labels: Optional[Iterator[Hashable]] = None,
    create_using: type[Graph] = Graph,
) -> Graph:
    """
    Generate a single topology representing a collection of disjoint linear
    chains according to a provided sequence of lengths and labelling scheme
    """
    return reduce(
        graph_union,
        path_graphs(
            chain_lengths=chain_lengths,
            node_labels=node_labels,
            create_using=create_using,
        ),
    )


def dendrimer_graph(
    coord_number: int = 3,
    coord_number_core: Optional[int] = None,
    num_generations: int = 5,
    label_attr_name: str = "label",
    core: Hashable = "core",
    shell_inner: Hashable = "branch",
    shell_outer: Hashable = "leaf",
) -> Graph:
    """
    Generate a graph which represents the connectivity of repeat units in
    a dendrimer formed via divergent (i.e. "from core outwards") synthesis

    Represented as a tree emanating from a distinguished "core" node and
    terminating at "leaves" all a topological distance <num_generations> from the core.
    Each branch off of the core is a Cayley graph with branching degree <coord_number>

    The nodes in the graph will have the following degrees:
    | Node type  | Degree            |
    | -----------+------------------ |
    | Core       | coord_number_core |
    | Leaf       | 1                 |
    | All others | coord_number      |

    Can also specify named labels for the core, branch, and leaf nodes

    Parameters
    ----------
    coord_number: int = 3
        The degree of every node which isn't the leaf or core

        Represents the number of bonds each generation of monomers
        can make to other repeat units when added to the tree,
        INCLUDING the bond back towards the core
    coord_number_core: Optional[int], default None
        The degree of the core node
        If None, will default to the value of <coord_number>
        Allows for configuration of the functionality of the core unit

        E.g. PAMAM dendrimers have 4-neighbored
        cores but only 3-neighbored branches
    num_generations: int, default 5
        The distance from the core to any leaf

        Corresponds to the number of synthetic steps needed in a
        divergent synthesis to achieve a molecule resembling this tree
    label_attr_name: str, default "label"
        The node attribute to which to bind node labels
        These labels identify the type of nodes in the trees
    core: Hashable, default "core"
        Label to use for the distinguished core node
    shell_inner: Hashable, default "branch"
        Label to use for all intermediate "branch"
        nodes between the core and the leaves
    shell_outer: Hashable, default "leaf",
        Label to use for all leaf nodes

    Returns
    -------
    dendrimer_graph : Graph
        An undirected Graph representing the dendrimer's repeat unit connectivity
    """
    if coord_number_core is None:
        coord_number_core = coord_number

    CORE_NODE: Hashable = 0
    dendr_graph = Graph()
    dendr_graph.add_node(CORE_NODE, **{label_attr_name: core})

    for branch_num in range(coord_number_core):
        branch_tree: DiGraph = balanced_tree(
            # incoming edge from center counts as one extra neigbhor
            coord_number - 1,
            # "extra" generation comes from connecting this subtree to the core
            num_generations - 1,
            # make directed initially to use in/out-degree info for orientation
            create_using=DiGraph,
        )
        relabel_nodes(
            branch_tree,
            mapping={node_idx: (branch_num, node_idx) for node_idx in branch_tree},
            copy=False,
        )

        for node in branch_tree.nodes:
            if branch_tree.out_degree(node) == 0:
                branch_tree.nodes[node][label_attr_name] = shell_outer
            else:
                branch_tree.nodes[node][label_attr_name] = shell_inner

            if branch_tree.in_degree(node) == 0:
                branch_core_node = node

        dendr_graph = compose(dendr_graph, Graph(branch_tree))
        dendr_graph.add_edge(CORE_NODE, branch_core_node)

    return dendr_graph
