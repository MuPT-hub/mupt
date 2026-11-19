"""Unit tests for LadderPolymerGenerator"""

import warnings

import networkx as nx
import pytest

from mupt.roles import has_SAAMR_roles
from mupt.polymers.exceptions import FaceError
from mupt.polymers.generators.ladder import LadderPolymerGenerator
from mupt.polymers.models import SequenceAlternating
from mupt.polymers.rules.junctions import FixedPairing, RandomFlip
from mupt.polymers.utils.lexicon import cap_key

LADDER = "[*:1]c1cc([*:3])c([*:4])cc1[*:2]"  # toy width-2 unit
LADDER_FACES = {"head": (1, 2), "tail": (3, 4)}

# PIM-1's repeat units: TTSBI (ports 1-4) and TFTPN (ports 5-8)
TTSBI = "C12(CC(C)(C)c3cc(O[*:3])c(O[*:4])cc31)CC(C)(C)c1cc(O[*:1])c(O[*:2])cc12"
TFTPN = "N#Cc1c([*:5])c([*:6])c(C#N)c([*:7])c1[*:8]"
PIM_FACES = {
    "TTSBI": {"head": (1, 2), "tail": (3, 4)},
    "TFTPN": {"head": (5, 6), "tail": (7, 8)},
}


def unit_keys(G: nx.Graph) -> list:
    return [G.nodes[node]["unit"] for node in G]


def toy_ladder(**kwargs) -> LadderPolymerGenerator:
    return LadderPolymerGenerator({"L": LADDER}, size=3, faces={"L": LADDER_FACES}, **kwargs)


def pim1(**kwargs) -> LadderPolymerGenerator:
    return LadderPolymerGenerator(
        {"TTSBI": TTSBI, "TFTPN": TFTPN},
        size=4,
        faces=PIM_FACES,
        sequence=SequenceAlternating(["TTSBI", "TFTPN"]),
        **kwargs,
    )


# Junctions
def test_junctions_have_two_bonds() -> None:
    G = toy_ladder().generate_topologies(1, seed=0)[0]
    assert all(len(bonds) == 2 for _, _, bonds in G.edges(data="bonds"))


def test_junctions_are_straight_by_default() -> None:
    G = toy_ladder().generate_topologies(1, seed=0)[0]
    assert G.edges[0, 1]["bonds"] == (((0, 3), (1, 1)), ((0, 4), (1, 2)))


def test_junction_rule_is_used() -> None:
    with pytest.warns(UserWarning, match="symmetry-equivalent"):  # the toy unit is symmetric
        generator = toy_ladder(junctions=RandomFlip(1.0))
    G = generator.generate_topologies(1, seed=0)[0]
    assert G.edges[0, 1]["bonds"] == (((0, 3), (1, 2)), ((0, 4), (1, 1)))  # flipped


def test_flipped_pim1_realizes_with_crossed_bonds() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TFTPN's symmetric faces warn; tested below
        generator = pim1(junctions=RandomFlip(1.0))
    G = generator.generate_topologies(1, seed=0)[0]
    chain = generator.realize(1, seed=0).children[0]
    units = list(chain.children)  # children are attached in node order

    for i in range(3):
        tail = PIM_FACES[G.nodes[i]["unit"][0]]["tail"]
        head = PIM_FACES[G.nodes[i + 1]["unit"][0]]["head"]

        # in the graph: tail strand 0 meets head strand 1, and vice versa
        assert G.edges[i, i + 1]["bonds"] == (((i, tail[0]), (i + 1, head[1])), ((i, tail[1]), (i + 1, head[0])))

        # in the realized chain: each upstream port's neighbor is the crossed downstream port
        downstream = {id(conn) for conn in units[i + 1].connections.connectors}
        across = {
            conn.label: conn.neighbor.label
            for conn in units[i].connections.connectors
            if conn.has_neighbor and id(conn.neighbor) in downstream
        }
        assert across == {tail[0]: head[1], tail[1]: head[0]}

    assert len(chain.connections.connectors_free) == 4  # both uncapped end faces
    assert has_SAAMR_roles(generator.realize(2, seed=0))


# End groups
def test_broadcast_cap_gives_one_cap_node_per_strand() -> None:
    generator = toy_ladder(end_groups="[H]*")
    G = generator.generate_topologies(1, seed=0)[0]

    assert unit_keys(G).count(cap_key("head")) == unit_keys(G).count(cap_key("tail")) == 2
    assert len(generator.realize(1, seed=0).children[0].connections.connectors_free) == 0


def test_multi_port_cap_gives_one_cap_node_per_end() -> None:
    generator = toy_ladder(end_groups="[*:1]OCO[*:2]")
    G = generator.generate_topologies(1, seed=0)[0]

    assert unit_keys(G).count(cap_key("head")) == unit_keys(G).count(cap_key("tail")) == 1
    assert len(generator.realize(1, seed=0).children[0].connections.connectors_free) == 0


def test_caps_need_one_port_or_one_per_strand() -> None:
    with pytest.raises(FaceError, match="3 ports"):
        toy_ladder(end_groups="[*:1]C([*:2])[*:3]")


# Symmetry warning
def test_flips_at_symmetric_faces_warn() -> None:
    with pytest.warns(UserWarning) as record:
        pim1(junctions=RandomFlip(0.5))
    messages = [str(warning.message) for warning in record]

    # TFTPN's faces are symmetric; TTSBI's catechol ports aren't
    assert any("'TFTPN'" in message for message in messages)
    assert not any("'TTSBI'" in message for message in messages)


def test_straight_junctions_never_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pim1(junctions=FixedPairing())


# What LadderPolymerGenerator accepts
def test_faces_are_required() -> None:
    with pytest.raises(TypeError):
        LadderPolymerGenerator({"L": LADDER}, size=3)


def test_single_bond_faces_point_to_the_linear_generator() -> None:
    with pytest.raises(FaceError, match="LinearPolymerGenerator"):
        LadderPolymerGenerator({"PE": "*[CH2:1][CH2:2]*"}, size=3, faces={"PE": {"head": 1, "tail": 2}})
