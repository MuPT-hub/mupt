"""Unit tests for LinearPolymerGenerator, and through it the shared chain generator"""

import networkx as nx
import numpy as np
import pytest

from mupt.roles import has_SAAMR_roles
from mupt.polymers.exceptions import FaceError, TopologyValidationError
from mupt.polymers.generators.linear import LinearPolymerGenerator
from mupt.polymers.models import (
    SequenceAlternating,
    TacticityBernoullian,
    SequenceExplicit,
    SequenceMayoLewis,
    SizeUniform,
    isotactic,
    syndiotactic,
)
from mupt.polymers.rules.junctions import RandomFlip
from mupt.polymers.utils.lexicon import cap_key
from mupt.polymers.utils.realize import universe_from_descriptive_graphs

PE = "*[CH2:1][CH2:2]*"
PP = "*[CH2:1][CH:2](C)*"
STYRENE = "*[CH2:1][CH:2](c1ccccc1)*"
MMA = "*[CH2:1][C:2](C)(C(=O)OC)*"
LADDER = "[*:1]c1cc([*:3])c([*:4])cc1[*:2]"  # toy width-2 unit
LADDER_FACES = {"head": (1, 2), "tail": (3, 4)}


def unit_keys(G: nx.Graph) -> list:
    return [G.nodes[node]["unit"] for node in G]


def same_graphs(first, second) -> bool:
    """Whether two lists of graphs have identical nodes, edges, and attributes"""
    return [(list(G.nodes(data=True)), list(G.edges(data=True))) for G in first] == [
        (list(G.nodes(data=True)), list(G.edges(data=True))) for G in second
    ]


# What a chain looks like
def test_chain_graph_is_a_head_to_tail_path() -> None:
    G = LinearPolymerGenerator({"PE": PE}, size=4).generate_topologies(1, seed=0)[0]

    assert unit_keys(G) == [("PE", None)] * 4
    assert [G.edges[i, i + 1]["bonds"] for i in range(3)] == [(((i, 2), (i + 1, 1)),) for i in range(3)]


def test_graphs_hold_only_what_realization_needs() -> None:
    G = LinearPolymerGenerator({"PE": PE}, size=4, end_groups="[H]*").generate_topologies(1, seed=0)[0]

    assert all(set(data) == {"unit"} for _, data in G.nodes(data=True))
    assert all(set(data) == {"bonds"} for _, _, data in G.edges(data=True))
    assert G.graph == {}


def test_size_is_coerced_and_required() -> None:
    assert LinearPolymerGenerator({"PE": PE}, size=3).generate_topology(np.random.default_rng(0)).number_of_nodes() == 3
    with pytest.raises(TypeError):
        LinearPolymerGenerator({"PE": PE}, size=None)


def test_default_sequence_uses_every_unit() -> None:
    graphs = LinearPolymerGenerator({"PE": PE, "PP": PP}, size=20).generate_topologies(5, seed=0)
    assert {key[0] for G in graphs for key in unit_keys(G)} == {"PE", "PP"}


# End groups
def test_capped_polystyrene() -> None:
    generator = LinearPolymerGenerator({"S": STYRENE}, size=5, end_groups="[H]*")
    G = generator.generate_topologies(1, seed=0)[0]
    assert unit_keys(G)[5:] == [cap_key("head"), cap_key("tail")]

    universe = generator.realize(3, seed=0)
    assert all(len(chain.children) == 5 + 2 for chain in universe.children)
    assert all(len(chain.connections.connectors_free) == 0 for chain in universe.children)
    assert has_SAAMR_roles(universe)


def test_uncapped_chains_keep_both_ends_free() -> None:
    universe = LinearPolymerGenerator({"PE": PE}, size=3).realize(2, seed=0)
    assert all(len(chain.connections.connectors_free) == 2 for chain in universe.children)


def test_one_capped_end() -> None:
    universe = LinearPolymerGenerator({"PE": PE}, size=3, end_groups={"tail": "[H]*"}).realize(1, seed=0)
    assert len(universe.children[0].connections.connectors_free) == 1  # the free head end


# Tacticity
@pytest.mark.parametrize("tacticity, alternating", [(isotactic(), False), (syndiotactic(), True)])
def test_tacticity_sets_unit_configs(tacticity, alternating) -> None:
    generator = LinearPolymerGenerator({"PP": PP}, size=6, tacticity=tacticity)
    configs = [key[1] for key in unit_keys(generator.generate_topologies(1, seed=0)[0])]

    assert set(configs) <= {1, -1}
    neighbours_differ = [a != b for a, b in zip(configs, configs[1:])]
    assert all(neighbours_differ) if alternating else not any(neighbours_differ)


def test_realized_residues_carry_their_configs() -> None:
    generator = LinearPolymerGenerator({"PP": PP}, size=6, tacticity=syndiotactic())
    graphs = generator.generate_topologies(2, seed=0)
    universe = universe_from_descriptive_graphs(graphs, generator.lexicon)

    for G, chain in zip(graphs, universe.children):
        # children are attached in node order
        assert [unit.metadata["config"] for unit in chain.children] == [G.nodes[n]["unit"][1] for n in G]


def test_tacticity_only_applies_to_tactic_units() -> None:
    generator = LinearPolymerGenerator(
        {"PE": PE, "PP": PP}, size=6, sequence=SequenceAlternating(["PE", "PP"]), tacticity=isotactic()
    )
    keys = unit_keys(generator.generate_topologies(1, seed=0)[0])

    assert generator.tactic_labels == {"PP"}
    assert all(config is None for label, config in keys if label == "PE")
    assert all(config in (1, -1) for label, config in keys if label == "PP")


def pp_configs(generator, n_chains: int) -> list[list]:
    """The PP configs of each generated chain, in chain order"""
    return [
        [config for label, config in unit_keys(G) if label == "PP"]
        for G in generator.generate_topologies(n_chains, seed=0)
    ]


def test_tactic_units_separated_by_non_tactic_ones_reset_by_default() -> None:
    # PP-PE-PP-PE-...: every PP is isolated, so isotactic chain-end control has no effect
    generator = LinearPolymerGenerator(
        {"PE": PE, "PP": PP}, size=20, sequence=SequenceAlternating(["PP", "PE"]), tacticity=isotactic()
    )
    pairs = [(a, b) for configs in pp_configs(generator, 100) for a, b in zip(configs, configs[1:])]

    same = sum(a == b for a, b in pairs) / len(pairs)
    assert same == pytest.approx(0.5, abs=0.06)  # ~3.4 sigma for 900 pairs


def test_tactic_units_carry_across_gaps_when_asked() -> None:
    generator = LinearPolymerGenerator(
        {"PE": PE, "PP": PP},
        size=20,
        sequence=SequenceAlternating(["PP", "PE"]),
        tacticity=isotactic(across_gaps="carry"),
    )
    assert all(len(set(configs)) == 1 for configs in pp_configs(generator, 10))


def test_tacticity_without_stereocenters_warns() -> None:
    with pytest.warns(UserWarning, match="no unit has backbone stereocenters"):
        LinearPolymerGenerator({"PE": PE}, size=3, tacticity=isotactic())


# Reproducibility
def test_explicit_sequence_and_seed_reproduce_chains() -> None:
    generator = LinearPolymerGenerator(
        {"PE": PE, "PP": PP}, size=4, sequence=SequenceExplicit(["PP", "PE", "PE", "PP"]), tacticity=isotactic()
    )
    assert same_graphs(generator.generate_topologies(3, seed=9), generator.generate_topologies(3, seed=9))
    assert [key[0] for key in unit_keys(generator.generate_topologies(1, seed=0)[0])] == ["PP", "PE", "PE", "PP"]


def test_seed_regression() -> None:
    # pins the random draw order (size, then sequence, then tacticity): if this changes,
    # every seeded result users have changes too, so update it only deliberately
    generator = LinearPolymerGenerator(
        {"S": STYRENE, "M": MMA},
        size=SizeUniform(4, 8),
        sequence=SequenceMayoLewis(feed={"S": 1, "M": 1}, reactivity={("S", "M"): 0.52, ("M", "S"): 0.46}),
        tacticity=TacticityBernoullian(0.6),
    )
    G = generator.generate_topologies(1, seed=0)[0]
    assert unit_keys(G) == [
        ("S", -1), ("S", 1), ("S", 1), ("M", -1), ("M", -1), ("S", 1), ("M", 1), ("S", -1),
    ]


# Hand-edited graphs
def test_hand_edits_are_realized() -> None:
    generator = LinearPolymerGenerator({"PE": PE, "PP": PP}, size=4, sequence=SequenceExplicit(["PE"] * 4), tacticity=isotactic())
    G = generator.generate_topologies(1, seed=0)[0]

    G.nodes[2]["unit"] = ("PP", 1)  # swap one unit for a tactic PP
    chain = universe_from_descriptive_graphs([G], generator.lexicon).children[0]
    assert [unit.metadata["unit_label"] for unit in chain.children] == ["PE", "PE", "PP", "PE"]


def test_corrupted_hand_edits_raise() -> None:
    generator = LinearPolymerGenerator({"PE": PE}, size=3)
    G = generator.generate_topologies(1, seed=0)[0]

    G.edges[0, 1]["bonds"] = (((0, 2), (1, 9)),)  # no port 9
    with pytest.raises(TopologyValidationError):
        universe_from_descriptive_graphs([G], generator.lexicon)


# End to end: a styrene-MMA terminal copolymer
def test_styrene_mma_terminal_copolymer() -> None:
    generator = LinearPolymerGenerator(
        {"S": STYRENE, "M": MMA},
        size=SizeUniform(10, 20),
        end_groups="[H]*",
        sequence=SequenceMayoLewis(feed={"S": 1, "M": 1}, reactivity={("S", "M"): 0.52, ("M", "S"): 0.46}),
        tacticity=TacticityBernoullian(0.6),
    )
    universe = generator.realize(4, seed=1)

    assert len(universe.children) == 4
    assert all(len(chain.connections.connectors_free) == 0 for chain in universe.children)
    labels = {unit.metadata["unit_label"] for chain in universe.children for unit in chain.children}
    assert {"S", "M"} <= labels
    assert has_SAAMR_roles(universe)


# What LinearPolymerGenerator accepts
def test_multi_bond_faces_point_to_the_ladder_generator() -> None:
    with pytest.raises(FaceError, match="LadderPolymerGenerator"):
        LinearPolymerGenerator({"L": LADDER}, size=3, faces={"L": LADDER_FACES})


def test_junction_rules_arent_accepted() -> None:
    with pytest.raises(TypeError):
        LinearPolymerGenerator({"PE": PE}, size=3, junctions=RandomFlip(0.5))


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(units={}), ValueError, "At least one"),
        (dict(units={"PE": PE}, faces={"PP": {"head": 1, "tail": 2}}), FaceError, "unknown unit"),
        (dict(units={"PE": PE}, faces={"PE": {"a": 1, "b": 2}}), FaceError, "missing face"),
        (dict(units={"PE": PE, "L": LADDER}, faces={"L": LADDER_FACES}), FaceError, "same width"),
        (dict(units={"PE": PE}, sequence=SequenceAlternating(["PE", "PP"])), ValueError, "not given in units"),
        (dict(units={"D": "*=[CH:1][CH:2]=*"}, end_groups="[H]*"), FaceError, "bond type"),
        (dict(units={"PE": PE}, end_groups={"middle": "[H]*"}), ValueError, "'head' and/or 'tail'"),
        (dict(units={"PE": PE}, side_groups={"PP": 5}), ValueError, "Side groups given for unknown"),
    ],
)
def test_constructor_errors(kwargs, error, message) -> None:
    kwargs.setdefault("size", 3)
    with pytest.raises(error, match=message):
        LinearPolymerGenerator(**kwargs)


# Named caps, and caps with free faces
ESTER_CAP = "[*:5]C(=O)[*:2]"  # port 2 bonds the chain, port 5 stays free


def ester_capped_ps(**kwargs) -> LinearPolymerGenerator:
    return LinearPolymerGenerator(
        {"S": STYRENE},
        size=4,
        end_groups={"head": "[H]*", "tail": {"ester": ESTER_CAP}},
        faces={"ester": {"cap": 2, "link": 5}},
        **kwargs,
    )


def test_named_cap_is_keyed_by_its_label() -> None:
    generator = ester_capped_ps()
    G = generator.generate_topologies(1, seed=0)[0]

    assert unit_keys(G)[-2:] == [cap_key("head"), ("ester", None)]
    assert generator.lexicon[("ester", None)].metadata["residue_name"] == "ester"


def test_named_cap_keeps_its_other_faces_free() -> None:
    chain = ester_capped_ps().realize(1, seed=0).children[0]
    free = chain.connections.connectors_free
    assert [conn.label for conn in free] == [5]  # the ester's link port, nothing else


def test_faces_and_free_sites_include_caps() -> None:
    generator = ester_capped_ps()
    assert generator.faces["ester"] == {"cap": (2,), "link": (5,)}
    assert generator.faces["S"] == {"head": (1,), "tail": (2,)}
    assert generator.possible_free_sites() == {("ester", "link")}  # head capped, tail exposes the link

    uncapped = LinearPolymerGenerator({"S": STYRENE, "M": MMA}, size=4)
    assert uncapped.possible_free_sites() == {("S", "head"), ("M", "head"), ("S", "tail"), ("M", "tail")}


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(end_groups={"tail": {"S": "[H]*"}}), ValueError, "already used"),
        (dict(end_groups={"tail": {"<tail_cap>": "[H]*"}}), ValueError, "reserved"),
        (dict(end_groups={"head": {"x": "[H]*"}, "tail": {"x": "[Br]*"}}), ValueError, "different caps"),
        (dict(end_groups={"tail": {"ester": ESTER_CAP}}, faces={"ester": {"link": 5, "other": 2}}), FaceError, "'cap'"),
        (dict(end_groups={"tail": {1: "[H]*"}}), TypeError, "str"),
        (dict(units={("S",): STYRENE}), TypeError, "str"),
    ],
)
def test_named_cap_errors(kwargs, error, message) -> None:
    kwargs.setdefault("units", {"S": STYRENE})
    with pytest.raises(error, match=message):
        LinearPolymerGenerator(size=3, **kwargs)
