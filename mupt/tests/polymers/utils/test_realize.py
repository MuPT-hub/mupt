"""Unit tests for realizing descriptive graphs as Primitives (no generators involved)"""

import networkx as nx
import pytest
from networkx import MultiGraph

from mupt.interfaces.smiles import primitive_from_smiles
from mupt.roles import has_SAAMR_roles
from mupt.polymers.exceptions import TopologyValidationError
from mupt.polymers.rules.faces import free_ports_by_label
from mupt.polymers.utils.realize import (
    primitive_from_descriptive_graph,
    universe_from_descriptive_graphs,
)

# repeat units, with ports labelled by map numbers
PE = "*[CH2:1][CH2:2]*"  # linear: head port 1, tail port 2
MIXED = "[*:1]=CC[*:2]"  # port 1 is a double bond, port 2 a single bond
TTSBI = "C12(CC(C)(C)c3cc(O[*:3])c(O[*:4])cc31)CC(C)(C)c1cc(O[*:1])c(O[*:2])cc12"  # ports 1-4
TFTPN = "N#Cc1c([*:5])c([*:6])c(C#N)c([*:7])c1[*:8]"  # ports 5-8


def template(smiles: str, label: str, embed: bool = False):
    """Lexicon template for a unit SMILES, coordinate-free unless asked for"""
    return primitive_from_smiles(smiles, label=label, embed_positions=embed, embed_random_seed=0)


@pytest.fixture(scope="module")
def lexicon() -> dict:
    return {
        "PE": template(PE, "PE"),
        "MIXED": template(MIXED, "MIXED"),
        "TTSBI": template(TTSBI, "TTSBI"),
        "TFTPN": template(TFTPN, "TFTPN"),
    }


def linear_graph(n_units: int, unit: str = "PE") -> nx.Graph:
    """Path graph of n_units, each bonding its tail port 2 to the next unit's head port 1"""
    G = nx.path_graph(n_units)
    nx.set_node_attributes(G, unit, "unit")
    for i in range(n_units - 1):
        G.edges[i, i + 1]["bonds"] = (((i, 2), (i + 1, 1)),)
    return G


def ladder_dimer(bonds) -> nx.Graph:
    """TTSBI (node 0) joined to TFTPN (node 1) by the given bonds"""
    G = nx.Graph()
    G.add_node(0, unit="TTSBI")
    G.add_node(1, unit="TFTPN")
    G.add_edge(0, 1, bonds=bonds)
    return G


def children_by_node(G: nx.Graph, chain) -> dict:
    """Pair each node with its realized unit (children are attached in node order)"""
    return dict(zip(G.nodes, chain.children))


def unit_level(chain):
    """Predicate selecting a chain's direct children, for cross_section"""
    return lambda prim: prim.parent is chain


# Building chains
def test_linear_trimer(lexicon) -> None:
    G = linear_graph(3)
    chain = primitive_from_descriptive_graph(G, lexicon)

    assert len(chain.children) == 3
    assert len(chain.connections.connectors_free) == 2  # the two uncapped ends
    assert chain.cross_section(unit_level(chain)).number_of_edges() == 2


def test_bonds_join_the_named_ports(lexicon) -> None:
    G = linear_graph(3)
    chain = primitive_from_descriptive_graph(G, lexicon)
    units = children_by_node(G, chain)

    # the free ports left over are node 0's head and node 2's tail
    free = set(chain.connections.connectors_free)
    assert free == {
        next(c for c in units[0].connections.connectors_free if c.label == 1),
        next(c for c in units[2].connections.connectors_free if c.label == 2),
    }


def test_single_unit_graph(lexicon) -> None:
    G = nx.Graph()
    G.add_node("only", unit="PE")
    chain = primitive_from_descriptive_graph(G, lexicon, label="lonely")

    assert chain.label == "lonely"
    assert len(chain.children) == 1
    assert len(chain.connections.connectors_free) == 2


def test_ladder_dimer_has_two_bonds_on_one_edge(lexicon) -> None:
    G = ladder_dimer((((0, 3), (1, 5)), ((0, 4), (1, 6))))
    chain = primitive_from_descriptive_graph(G, lexicon)

    # TTSBI keeps ports 1, 2 and TFTPN keeps ports 7, 8 free
    assert len(chain.connections.connectors_free) == 4

    # two bonds between the same pair of units turn the cross section into a MultiGraph
    section = chain.cross_section(unit_level(chain))
    assert isinstance(section, MultiGraph)
    assert section.number_of_edges() == 2


@pytest.mark.parametrize(
    "bonds, expected",
    [
        ((((0, 3), (1, 5)), ((0, 4), (1, 6))), {3: 5, 4: 6}),  # straight junction
        ((((0, 3), (1, 6)), ((0, 4), (1, 5))), {3: 6, 4: 5}),  # flipped junction
    ],
)
def test_ladder_pairing_follows_the_bonds(lexicon, bonds, expected) -> None:
    G = ladder_dimer(bonds)
    units = children_by_node(G, primitive_from_descriptive_graph(G, lexicon))

    # TTSBI's connectors bonded across the junction (skipping its internal, intra-unit bonds),
    # mapped to the label of the TFTPN port each one reached
    tftpn_connectors = {id(conn) for conn in units[1].connections.connectors}
    across = {
        conn.label: conn.neighbor.label
        for conn in units[0].connections.connectors
        if conn.has_neighbor and id(conn.neighbor) in tftpn_connectors
    }
    assert across == expected


def test_embedded_templates_bond_without_alignment(lexicon) -> None:
    # generators supply embedded templates; bonding must not require them to be aligned
    embedded = {"PE": template(PE, "PE", embed=True)}
    chain = primitive_from_descriptive_graph(linear_graph(4), embedded)
    assert len(chain.connections.connectors_free) == 2


def test_templates_are_left_untouched(lexicon) -> None:
    primitive_from_descriptive_graph(linear_graph(3), lexicon)
    assert set(free_ports_by_label(lexicon["PE"])) == {1, 2}  # copies were bonded, not the template


def test_metadata_comes_from_templates(lexicon) -> None:
    labelled = {"PE": template(PE, "PE")}
    labelled["PE"].metadata["residue_name"] = "PE"

    G = linear_graph(2)
    G.nodes[0]["note"] = "extra attributes are ignored"
    chain = primitive_from_descriptive_graph(G, labelled)

    assert all(unit.metadata["residue_name"] == "PE" for unit in chain.children)
    assert all("note" not in unit.metadata for unit in chain.children)


def test_custom_attribute_names(lexicon) -> None:
    G = nx.path_graph(2)
    G.nodes[0]["template"] = G.nodes[1]["template"] = "PE"
    G.edges[0, 1]["links"] = (((0, 2), (1, 1)),)

    chain = primitive_from_descriptive_graph(G, lexicon, unit_attr="template", bonds_attr="links")
    assert len(chain.connections.connectors_free) == 2


# Universes
def test_universe_has_one_chain_per_graph(lexicon) -> None:
    universe = universe_from_descriptive_graphs([linear_graph(3), linear_graph(5)], lexicon)

    assert [chain.label for chain in universe.children] == ["chain_0", "chain_1"]
    assert [len(chain.children) for chain in universe.children] == [3, 5]
    assert has_SAAMR_roles(universe)


# Validation errors
def _with_edge_bonds(G: nx.Graph, bonds) -> nx.Graph:
    G.edges[0, 1]["bonds"] = bonds
    return G


def _graph_missing_unit() -> nx.Graph:
    G = linear_graph(2)
    del G.nodes[1]["unit"]
    return G


def _graph_unknown_unit() -> nx.Graph:
    G = linear_graph(2)
    G.nodes[1]["unit"] = "NYLON"
    return G


def _graph_missing_bonds() -> nx.Graph:
    G = linear_graph(2)
    del G.edges[0, 1]["bonds"]
    return G


def _graph_bond_off_its_edge() -> nx.Graph:
    G = linear_graph(3)
    G.edges[0, 1]["bonds"] = (((0, 2), (2, 1)),)  # joins nodes 0 and 2, stored on edge (0, 1)
    return G


def _graph_port_used_twice() -> nx.Graph:
    G = linear_graph(3)
    G.edges[1, 2]["bonds"] = (((1, 1), (2, 1)),)  # node 1's port 1 is already used by edge (0, 1)
    return G


def _graph_missing_port() -> nx.Graph:
    return _with_edge_bonds(linear_graph(2), (((0, 2), (1, 9)),))


def _graph_bondtype_mismatch() -> nx.Graph:
    G = linear_graph(2)
    G.nodes[1]["unit"] = "MIXED"  # its port 1 is a double bond; PE's port 2 is single
    return G


def _graph_malformed_bond() -> nx.Graph:
    return _with_edge_bonds(linear_graph(2), ((0, 2, 1, 1),))


def _graph_self_loop() -> nx.Graph:
    G = linear_graph(1)
    G.add_edge(0, 0, bonds=(((0, 1), (0, 2)),))
    return G


@pytest.mark.parametrize(
    "build_graph, message",
    [
        (_graph_missing_unit, "no 'unit'"),
        (_graph_unknown_unit, "isn't in the lexicon"),
        (_graph_missing_bonds, "no 'bonds'"),
        (_graph_bond_off_its_edge, "joins nodes"),
        (_graph_port_used_twice, "again on edge"),
        (_graph_missing_port, "names port 9"),
        (_graph_bondtype_mismatch, "different bond types"),
        (_graph_malformed_bond, "isn't of the form"),
        (_graph_self_loop, "self-loop"),
    ],
)
def test_invalid_graphs_raise(lexicon, build_graph, message) -> None:
    with pytest.raises(TopologyValidationError, match=message):
        primitive_from_descriptive_graph(build_graph(), lexicon)
