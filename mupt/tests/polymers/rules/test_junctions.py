"""Unit tests for junction rules, which write the bonds of each edge in a descriptive graph"""

import warnings

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from mupt.chemistry.rdkit.sanitization import sanitized_mol
from mupt.chemistry.smiles import DEFAULT_SMILES_READ_PARAMS
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.polymers.exceptions import FaceError, TopologyValidationError
from mupt.polymers.rules.faces import LINEAR_FACES, face_anchor_indices
from mupt.polymers.rules.junctions import (
    FixedPairing,
    JunctionRule,
    RandomFlip,
    anchors_symmetry_equivalent,
    faces_of_node,
    is_permutation,
    junction_widths,
)
from mupt.polymers.utils.realize import primitive_from_descriptive_graph

# face specs keyed by unit label
LADDER_FACES = {"head": (1, 2), "tail": (3, 4)}
FACES = {"PE": dict(LINEAR_FACES), "U": LADDER_FACES}


def chain_graph(n_units: int, label: str = "U") -> nx.Graph:
    """Path graph whose nodes carry generated-style unit keys (label, config)"""
    G = nx.path_graph(n_units)
    nx.set_node_attributes(G, (label, None), "unit")
    return G


def chain_junctions(n_units: int) -> list[tuple]:
    """Head-to-tail junctions along a chain, in growth order"""
    return [(i, "tail", i + 1, "head") for i in range(n_units - 1)]


def rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


# Helpers
def test_is_permutation() -> None:
    assert is_permutation((1, 0), 2)
    assert not is_permutation((0, 0), 2)
    assert not is_permutation((0,), 2)


def test_faces_of_node_handles_tuple_and_plain_keys() -> None:
    G = nx.Graph()
    G.add_node("a", unit=("U", 1))  # generated key: looked up by its label
    G.add_node("b", unit="U")  # hand-written key: looked up as is
    G.add_node("c", unit="NYLON")

    assert faces_of_node(G, "a", FACES) == faces_of_node(G, "b", FACES) == LADDER_FACES
    with pytest.raises(FaceError, match="No faces"):
        faces_of_node(G, "c", FACES)


def test_junction_widths_reject_mismatched_faces() -> None:
    G = chain_graph(2)
    G.nodes[1]["unit"] = ("PE", None)  # width-1 faces meeting a width-2 face

    assert junction_widths(chain_graph(3), chain_junctions(3), FACES) == [2, 2]
    with pytest.raises(FaceError, match="Can't join"):
        junction_widths(G, chain_junctions(2), FACES)


# FixedPairing
@pytest.mark.parametrize("label, width", [("PE", 1), ("U", 2)])
def test_fixed_pairing_is_straight_and_draws_nothing(label, width) -> None:
    rng = np.random.default_rng(0)
    before = rng_state(rng)

    pairings = FixedPairing().pairings(chain_graph(4, label), chain_junctions(4), FACES, rng)
    assert pairings == [tuple(range(width))] * 3
    assert rng_state(rng) == before


# RandomFlip
def test_random_flip_fraction() -> None:
    n_units = 10_001
    pairings = RandomFlip(0.3).pairings(
        chain_graph(n_units), chain_junctions(n_units), FACES, np.random.default_rng(1)
    )
    flipped = sum(pairing == (1, 0) for pairing in pairings) / len(pairings)
    assert flipped == pytest.approx(0.3, abs=0.02)  # ~4.4 sigma at n = 10^4


def test_random_flip_draws_once_per_junction_at_any_width() -> None:
    # widths 1 and 2 must leave the random stream in the same state as 5 plain draws
    states = []
    for label in ("PE", "U"):
        rng = np.random.default_rng(7)
        RandomFlip(0.5).pairings(chain_graph(6, label), chain_junctions(6), FACES, rng)
        states.append(rng_state(rng))

    reference = np.random.default_rng(7)
    reference.random(5)
    assert states[0] == states[1] == rng_state(reference)


def test_random_flip_on_width_one_is_always_straight() -> None:
    pairings = RandomFlip(1.0).pairings(chain_graph(3, "PE"), chain_junctions(3), FACES, np.random.default_rng(0))
    assert pairings == [(0,), (0,)]


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_random_flip_rejects_bad_probabilities(p) -> None:
    with pytest.raises(ValueError):
        RandomFlip(p)


def test_random_flip_warns_when_nothing_can_flip() -> None:
    with pytest.warns(UserWarning, match="no effect"):
        RandomFlip(0.5).validate({"PE": LINEAR_FACES})

    # width-2 faces can flip, so there must be no warning (any warning becomes an error here)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RandomFlip(0.5).validate({"U": LADDER_FACES})


# assign
def test_assign_writes_straight_bonds() -> None:
    G = chain_graph(3)
    FixedPairing().assign(G, chain_junctions(3), FACES, np.random.default_rng(0))

    assert G.edges[0, 1] == {"bonds": (((0, 3), (1, 1)), ((0, 4), (1, 2)))}  # nothing but bonds
    assert G.edges[1, 2]["bonds"] == (((1, 3), (2, 1)), ((1, 4), (2, 2)))


def test_assign_writes_flipped_bonds() -> None:
    G = chain_graph(2)
    RandomFlip(1.0).assign(G, chain_junctions(2), FACES, np.random.default_rng(0))
    assert G.edges[0, 1]["bonds"] == (((0, 3), (1, 2)), ((0, 4), (1, 1)))


def test_assign_rejects_junctions_off_the_graph() -> None:
    with pytest.raises(TopologyValidationError, match="isn't an edge"):
        FixedPairing().assign(chain_graph(3), [(0, "tail", 2, "head")], FACES, np.random.default_rng(0))


class _BrokenRule(JunctionRule):
    """A rule returning an invalid pairing, to check assign's own guard"""

    def pairings(self, G, junctions, faces, rng):
        return [(0, 0)] * len(junctions)


def test_assign_rejects_invalid_pairings() -> None:
    with pytest.raises(ValueError, match="gave pairing"):
        _BrokenRule().assign(chain_graph(2), chain_junctions(2), FACES, np.random.default_rng(0))


# Extending the framework
class AlternatingFlip(JunctionRule):
    """Example extension: flip every other junction, overriding pairings() only"""

    def pairings(self, G, junctions, faces, rng):
        widths = junction_widths(G, junctions, faces)
        return [
            tuple(reversed(range(width))) if (i % 2) else tuple(range(width))
            for i, width in enumerate(widths)
        ]


def test_subclass_overriding_only_pairings() -> None:
    G = chain_graph(4)
    AlternatingFlip().assign(G, chain_junctions(4), FACES, np.random.default_rng(0))

    assert G.edges[0, 1]["bonds"] == (((0, 3), (1, 1)), ((0, 4), (1, 2)))  # straight
    assert G.edges[1, 2]["bonds"] == (((1, 3), (2, 2)), ((1, 4), (2, 1)))  # flipped
    assert G.edges[2, 3]["bonds"] == (((2, 3), (3, 1)), ((2, 4), (3, 2)))  # straight


# Junction rules and realization together
TTSBI = "C12(CC(C)(C)c3cc(O[*:3])c(O[*:4])cc31)CC(C)(C)c1cc(O[*:1])c(O[*:2])cc12"
TFTPN = "N#Cc1c([*:5])c([*:6])c(C#N)c([*:7])c1[*:8]"
PIM_FACES = {
    "TTSBI": {"head": (1, 2), "tail": (3, 4)},
    "TFTPN": {"head": (5, 6), "tail": (7, 8)},
}


def test_assigned_bonds_realize() -> None:
    lexicon = {
        ("TTSBI", None): primitive_from_smiles(TTSBI, embed_positions=False),
        ("TFTPN", None): primitive_from_smiles(TFTPN, embed_positions=False),
    }
    G = nx.path_graph(3)
    for node, label in enumerate(("TTSBI", "TFTPN", "TTSBI")):
        G.nodes[node]["unit"] = (label, None)

    RandomFlip(1.0).assign(G, chain_junctions(3), PIM_FACES, np.random.default_rng(0))
    chain = primitive_from_descriptive_graph(G, lexicon)

    # 2 bonds per junction; each end keeps its outer face (2 ports) free
    assert len(chain.connections.connectors_free) == 4

    # the first junction is flipped: TTSBI port 3 reached TFTPN port 6, and 4 reached 5
    ttsbi, tftpn = chain.children[0], chain.children[1]
    tftpn_connectors = {id(conn) for conn in tftpn.connections.connectors}
    across = {
        conn.label: conn.neighbor.label
        for conn in ttsbi.connections.connectors
        if conn.has_neighbor and id(conn.neighbor) in tftpn_connectors
    }
    assert across == {3: 6, 4: 5}


# Symmetry heuristic
def mol_from(smiles: str) -> Chem.Mol:
    return sanitized_mol(Chem.MolFromSmiles(smiles, DEFAULT_SMILES_READ_PARAMS), add_Hs=True)


@pytest.mark.parametrize(
    "smiles, face, expected",
    [
        (TFTPN, (5, 6), True),  # TFTPN is symmetric: flipping its head face changes nothing
        ("c1([*:1])c([*:2])c(C)ccc1", (1, 2), False),  # a methyl next to port 2 breaks the symmetry
    ],
)
def test_anchors_symmetry_equivalent(smiles, face, expected) -> None:
    mol = mol_from(smiles)
    anchors = face_anchor_indices(mol, {"head": face}, "head")
    assert anchors_symmetry_equivalent(mol, anchors) is expected
