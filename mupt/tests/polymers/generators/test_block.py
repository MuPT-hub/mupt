"""Unit tests for BlockCopolymerGenerator, and through it CompositeGenerator"""

import warnings

import numpy as np
import pytest

from mupt.roles import has_SAAMR_roles
from mupt.polymers.exceptions import CompatibilityError
from mupt.polymers.generators.block import BlockCopolymerGenerator
from mupt.polymers.generators.ladder import LadderPolymerGenerator
from mupt.polymers.generators.linear import LinearPolymerGenerator
from mupt.polymers.models import SizeUniform, isotactic, syndiotactic
from mupt.polymers.rules.compatibility import CompatibleSites
from mupt.polymers.rules.junctions import RandomFlip

STYRENE = "*[CH2:1][CH:2](c1ccccc1)*"
MMA = "*[CH2:1][C:2](C)(C(=O)OC)*"
PE = "*[CH2:1][CH2:2]*"
LADDER = "[*:1]c1cc([*:3])c([*:4])cc1[*:2]"
LADDER_FACES = {"head": (1, 2), "tail": (3, 4)}


def unit_keys(G) -> list:
    return [G.nodes[node]["unit"] for node in G]


def ps(**kwargs) -> LinearPolymerGenerator:
    return LinearPolymerGenerator({"S": STYRENE}, size=kwargs.pop("size", 4), tacticity=isotactic(), **kwargs)


def pmma(**kwargs) -> LinearPolymerGenerator:
    return LinearPolymerGenerator({"M": MMA}, size=kwargs.pop("size", 3), tacticity=syndiotactic(), **kwargs)


def diblock() -> BlockCopolymerGenerator:
    return BlockCopolymerGenerator(
        {"PS": ps(end_groups={"head": "[H]*"}), "PMMA": pmma(end_groups={"tail": "[H]*"})},
        blocks=("PS", "PMMA"),
    )


# PS-b-PMMA
def test_diblock_joins_last_tail_to_first_head() -> None:
    G = diblock().generate_topologies(1, seed=0)[0]

    assert G.number_of_nodes() == 4 + 1 + 3 + 1  # blocks plus one cap each
    assert G.edges[3, 5]["bonds"] == (((3, 2), (5, 1)),)  # PS's last unit (3) -> PMMA's first (5; 4 is PS's cap)


def test_keys_are_namespaced_and_residues_keep_their_labels() -> None:
    generator = diblock()
    keys = unit_keys(generator.generate_topologies(1, seed=0)[0])
    assert {key[0] for key in keys} == {("PS", "S"), ("PS", "<head_cap>"), ("PMMA", "M"), ("PMMA", "<tail_cap>")}

    chain = generator.realize(1, seed=0).children[0]
    units = [unit for unit in chain.children if unit.metadata["unit_label"] in ("S", "M")]
    assert [unit.metadata["unit_label"] for unit in units] == ["S"] * 4 + ["M"] * 3
    assert {unit.metadata["component"] for unit in units} == {("PS",), ("PMMA",)}


def test_capped_diblock_realizes_closed() -> None:
    universe = diblock().realize(3, seed=0)
    assert all(len(chain.connections.connectors_free) == 0 for chain in universe.children)
    assert has_SAAMR_roles(universe)


def test_each_block_keeps_its_own_tacticity() -> None:
    G = diblock().generate_topologies(1, seed=0)[0]
    s_configs = [key[1] for key in unit_keys(G) if key[0] == ("PS", "S")]
    m_configs = [key[1] for key in unit_keys(G) if key[0] == ("PMMA", "M")]
    assert len(set(s_configs)) == 1  # isotactic
    assert all(a != b for a, b in zip(m_configs, m_configs[1:]))  # syndiotactic


def test_blocks_draw_in_order_and_seeds_reproduce() -> None:
    generator = BlockCopolymerGenerator({"PS": ps(size=SizeUniform(2, 9)), "PMMA": pmma()}, blocks=("PS", "PMMA"))
    G = generator.generate_topology(np.random.default_rng(5))

    expected = ps(size=SizeUniform(2, 9)).generate_topology(np.random.default_rng(5))  # the first block, alone
    assert [key[1] for key in unit_keys(G)[: expected.number_of_nodes()]] == [key[1] for key in unit_keys(expected)]

    first, second = generator.generate_topologies(3, seed=1), generator.generate_topologies(3, seed=1)
    assert [unit_keys(G) for G in first] == [unit_keys(G) for G in second]


# Other arrangements
def test_aba_triblock_reuses_a_component() -> None:
    generator = BlockCopolymerGenerator({"A": ps(size=SizeUniform(2, 9)), "B": pmma()}, blocks=("A", "B", "A"))
    G = generator.generate_topologies(1, seed=2)[0]

    labels = [key[0] for key in unit_keys(G)]
    n_first = labels.index(("B", "M"))
    n_last = len(labels) - n_first - 3
    assert labels == [("A", "S")] * n_first + [("B", "M")] * 3 + [("A", "S")] * n_last
    assert G.number_of_edges() == G.number_of_nodes() - 1  # one unbranched chain


def test_nested_block_copolymers_have_flat_keys() -> None:
    inner = BlockCopolymerGenerator({"A": ps(size=2), "B": pmma(size=2)}, blocks=("A", "B"))
    outer = BlockCopolymerGenerator({"AB": inner, "E": LinearPolymerGenerator({"PE": PE}, size=2)}, blocks=("AB", "E"))

    labels = [key[0] for key in unit_keys(outer.generate_topologies(1, seed=0)[0])]
    assert labels == [("AB", "A", "S")] * 2 + [("AB", "B", "M")] * 2 + [("E", "PE")] * 2
    assert outer.lexicon[(("AB", "A", "S"), 1)].metadata["component"] == ("AB", "A")


def test_ladder_blocks_with_flips_at_the_joins() -> None:
    ladder = LadderPolymerGenerator({"L": LADDER}, size=2, faces={"L": LADDER_FACES})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # RandomFlip on width-2 faces is fine; nothing to warn about here
        generator = BlockCopolymerGenerator({"X": ladder, "Y": ladder}, blocks=("X", "Y"), junctions=RandomFlip(1.0))

    G = generator.generate_topologies(1, seed=0)[0]
    assert G.edges[1, 2]["bonds"] == (((1, 3), (2, 2)), ((1, 4), (2, 1)))  # flipped at the join
    assert G.edges[0, 1]["bonds"] == (((0, 3), (1, 1)), ((0, 4), (1, 2)))  # inside a block: the block's own rule


def test_bare_units_can_be_blocks_with_an_explicit_rule() -> None:
    components = {"A": ps(size=2), "link": "[*:1][Si](C)(C)[*:2]"}  # a bare unit: faces named by port, 1 and 2

    # head_to_tail can't join it, which is caught at construction
    with pytest.raises(CompatibilityError, match="can't join"):
        BlockCopolymerGenerator(components, blocks=("A", "link", "A"))

    rule = CompatibleSites((("A", "tail"), ("link", 1)), (("link", 2), ("A", "head")))
    generator = BlockCopolymerGenerator(components, blocks=("A", "link", "A"), compatibility=rule)
    G = generator.generate_topologies(1, seed=0)[0]

    assert G.nodes[2]["unit"] == (("link", "unit"), None)
    assert G.edges[1, 2]["bonds"] == (((1, 2), (2, 1)),)
    assert G.edges[2, 3]["bonds"] == (((2, 2), (3, 1)),)


# Errors
@pytest.mark.parametrize(
    "components, blocks, error, message",
    [
        ({"A": ps()}, ("A",), ValueError, "at least two"),
        ({"A": ps()}, ("A", "B"), ValueError, "aren't among"),
        ({"A": ps(end_groups={"tail": "[H]*"}), "B": pmma()}, ("A", "B"), CompatibilityError, "can't join"),
        ({"A": ps(), "B": LadderPolymerGenerator({"L": LADDER}, size=2, faces={"L": LADDER_FACES})}, ("A", "B"),
         CompatibilityError, "can't join"),  # width 1 next to width 2
        ({1: ps(), "B": pmma()}, (1, "B"), TypeError, "str"),
    ],
)
def test_constructor_errors(components, blocks, error, message) -> None:
    with pytest.raises(error, match=message):
        BlockCopolymerGenerator(components, blocks=blocks)


def test_unused_components_warn() -> None:
    with pytest.warns(UserWarning, match="never used"):
        BlockCopolymerGenerator({"A": ps(), "B": pmma(), "C": pmma()}, blocks=("A", "B"))
