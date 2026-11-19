"""Unit tests for UnitGenerator"""

import pytest

from mupt.polymers.generators.unit import UnitGenerator

CORE = "[*:1]C([*:2])([*:3])[*:4]"


def test_default_faces_are_one_per_port() -> None:
    generator = UnitGenerator(CORE)
    assert generator.faces == {"unit": {1: (1,), 2: (2,), 3: (3,), 4: (4,)}}
    assert generator.possible_free_sites() == {("unit", port) for port in (1, 2, 3, 4)}


def test_explicit_faces_and_label() -> None:
    generator = UnitGenerator(CORE, label="hub", faces={"a": (1, 2), "b": (3, 4)})
    assert generator.faces == {"hub": {"a": (1, 2), "b": (3, 4)}}
    assert list(generator.lexicon) == [("hub", None)]


def test_one_node_and_no_random_draws() -> None:
    G = UnitGenerator(CORE).generate_topology(None)  # None would fail if a number were drawn
    assert list(G.nodes(data="unit")) == [(0, ("unit", None))]
    assert G.number_of_edges() == 0


def test_labels_must_be_str() -> None:
    with pytest.raises(TypeError, match="str"):
        UnitGenerator(CORE, label=1)
