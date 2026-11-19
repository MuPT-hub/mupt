"""Unit tests for the generic TopologyGenerator interface (chain generators: test_linear, test_ladder)"""

import networkx as nx
import numpy as np
import pytest

from mupt.interfaces.smiles import primitive_from_smiles
from mupt.roles import has_SAAMR_roles
from mupt.polymers.exceptions import TopologyValidationError
from mupt.polymers.generators.base import TopologyGenerator
from mupt.polymers.utils.realize import combine_universes, validate_descriptive_graph

PE = "*[CH2:1][CH2:2]*"


# A minimal generator, to test TopologyGenerator on its own
class RandomLengthPE(TopologyGenerator):
    """Polyethylene chains of 1-5 units, built by hand"""

    def __init__(self) -> None:
        self._lexicon = {"PE": primitive_from_smiles(PE, label="PE", embed_positions=False)}

    @property
    def lexicon(self):
        return self._lexicon

    @property
    def faces(self):
        return {"PE": {"head": (1,), "tail": (2,)}}

    def generate_topology(self, rng):
        G = nx.path_graph(int(rng.integers(1, 6)))
        nx.set_node_attributes(G, "PE", "unit")
        for i in range(G.number_of_nodes() - 1):
            G.edges[i, i + 1]["bonds"] = (((i, 2), (i + 1, 1)),)
        return G


def sizes(graphs) -> list[int]:
    return [G.number_of_nodes() for G in graphs]


# TopologyGenerator
def test_generate_topologies_makes_n_graphs() -> None:
    generator = RandomLengthPE()
    assert len(generator.generate_topologies(7, seed=0)) == 7
    assert generator.generate_topologies(0, seed=0) == []

    with pytest.raises(ValueError):
        generator.generate_topologies(-1)


def test_seeds_reproduce_batches() -> None:
    generator = RandomLengthPE()
    assert sizes(generator.generate_topologies(20, seed=3)) == sizes(generator.generate_topologies(20, seed=3))
    assert sizes(generator.generate_topologies(20, seed=3)) != sizes(generator.generate_topologies(20, seed=4))


def test_realize_builds_a_universe() -> None:
    universe = RandomLengthPE().realize(4, seed=0, label="pe_melt")

    assert universe.label == "pe_melt"
    assert [chain.label for chain in universe.children] == [f"chain_{i}" for i in range(4)]
    assert has_SAAMR_roles(universe)


def test_realize_into_a_universe_returns_a_new_one() -> None:
    generator = RandomLengthPE()
    first = generator.realize(3, seed=0)
    blend = generator.realize(2, seed=1, universe=first)

    assert len(first.children) == 3  # the given universe is untouched
    assert len(blend.children) == 5
    assert [chain.label for chain in blend.children] == [f"chain_{i}" for i in range(5)]
    assert has_SAAMR_roles(blend)


def test_combine_universes_keeps_order_and_bonds() -> None:
    generator = RandomLengthPE()
    universes = [generator.realize(2, seed=seed) for seed in range(3)]
    combined = combine_universes(*universes)

    originals = [chain for universe in universes for chain in universe.children]
    assert [len(chain.children) for chain in combined.children] == [len(chain.children) for chain in originals]
    assert all(len(chain.connections.connectors_free) == 2 for chain in combined.children)  # bonds copied


def test_validate_descriptive_graph_is_public() -> None:
    generator = RandomLengthPE()
    G = generator.generate_topology(np.random.default_rng(0))
    validate_descriptive_graph(G, generator.lexicon)

    G.nodes[0]["unit"] = "NYLON"
    with pytest.raises(TopologyValidationError):
        validate_descriptive_graph(G, generator.lexicon)
