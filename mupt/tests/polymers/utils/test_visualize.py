"""Unit tests for the quick plots of unit and component graphs"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

from mupt.polymers.generators.block import BlockCopolymerGenerator
from mupt.polymers.generators.ladder import LadderPolymerGenerator
from mupt.polymers.generators.linear import LinearPolymerGenerator
from mupt.polymers.generators.star import StarPolymerGenerator
from mupt.polymers.utils.compose import flatten_component_graph
from mupt.polymers.utils.visualize import _join_text, draw_component_graph, draw_topology

STYRENE = "*[CH2:1][CH:2](c1ccccc1)*"
PE = "*[CH2:1][CH2:2]*"
LADDER = "[*:1]c1cc([*:3])c([*:4])cc1[*:2]"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def diblock() -> BlockCopolymerGenerator:
    return BlockCopolymerGenerator(
        {"PS": LinearPolymerGenerator({"S": STYRENE}, size=3, end_groups={"head": "[H]*"}),
         "PE": LinearPolymerGenerator({"PE": PE}, size=2)},
        blocks=("PS", "PE"),
    )


def star() -> StarPolymerGenerator:
    arm = LinearPolymerGenerator({"PE": PE}, size=2, end_groups={"tail": "[H]*"})
    return StarPolymerGenerator({"hub": "[*:1]C([*:2])[*:3]", "arm": arm}, core="hub", arms={(1, 2, 3): "arm"})


# Component graphs
def test_resolved_joins_name_faces_or_ports() -> None:
    generator = diblock()
    C = generator.generate_component_graph(np.random.default_rng(0))

    assert _join_text(C, 0, 1, generator.faces) == "tail → head"
    assert _join_text(C, 0, 1, None) == "2 → 1"  # ports, without faces
    assert _join_text(C, 1, 0, generator.faces) == "tail → head"  # upstream first, either way round


def test_unresolved_joins_show_anchors() -> None:
    C = star().component_graph(np.random.default_rng(0))  # the core is resolved, the joins aren't
    assert {_join_text(C, *edge, None) for edge in C.edges} == {"1@0 → any", "2@0 → any", "3@0 → any"}


def test_multi_strand_joins_show_each_face_pair_once() -> None:
    ladder = LadderPolymerGenerator({"L": LADDER}, size=2, faces={"L": {"head": (1, 2), "tail": (3, 4)}})
    generator = BlockCopolymerGenerator({"X": ladder, "Y": ladder}, blocks=("X", "Y"))
    C = generator.generate_component_graph(np.random.default_rng(0))

    assert _join_text(C, 0, 1, generator.faces) == "tail → head"
    assert _join_text(C, 0, 1, None) == "3 → 1\n4 → 2"


@pytest.mark.parametrize("make", [diblock, star])
def test_component_graphs_draw(make) -> None:
    generator = make()
    C = generator.generate_component_graph(np.random.default_rng(0))
    assert draw_component_graph(C, faces=generator.faces) is not None
    assert draw_component_graph(generator.component_graph(np.random.default_rng(0))) is not None


# Unit graphs
def test_unit_graphs_draw_and_reject_component_graphs() -> None:
    C = diblock().generate_component_graph(np.random.default_rng(0))
    assert draw_topology(flatten_component_graph(C), show_ports=True) is not None

    with pytest.raises(TypeError, match="draw_component_graph"):
        draw_topology(C)
    with pytest.raises(TypeError, match="draw_topology"):
        draw_component_graph(flatten_component_graph(C))
