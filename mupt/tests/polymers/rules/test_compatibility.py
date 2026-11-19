"""Unit tests for port compatibility: free faces, site patterns, and matching"""

import networkx as nx
import numpy as np
import pytest

from mupt.polymers.exceptions import CompatibilityError
from mupt.polymers.rules.compatibility import (
    ANY,
    CompatibleSites,
    free_faces,
    head_to_tail,
    resolve_pattern,
    site_of,
)

LINEAR = {"head": (1,), "tail": (2,)}
LADDER = {"head": (1, 2), "tail": (3, 4)}


def chain(keys, faces, bonded=True) -> nx.Graph:
    """A hand-built path graph of the given unit keys, tail-to-head bonded"""
    G = nx.path_graph(len(keys))
    nx.set_node_attributes(G, dict(enumerate(keys)), "unit")
    for i in range(len(keys) - 1):
        tail = faces[keys[i][0]]["tail"]
        head = faces[keys[i + 1][0]]["head"]
        G.edges[i, i + 1]["bonds"] = tuple(((i, t), (i + 1, h)) for t, h in zip(tail, head)) if bonded else ()
    return G


# Free faces
def test_free_faces_of_a_chain_are_its_ends() -> None:
    faces = {"S": LINEAR, "M": LINEAR}
    G = chain([("S", 1), ("M", None), ("S", -1)], faces)
    assert free_faces(G, faces) == [(0, "head"), (2, "tail")]
    assert site_of(G, (2, "tail"), faces) == ("S", "tail")


def test_capped_ends_have_no_free_faces() -> None:
    faces = {"S": LINEAR, "<cap>": {"cap": (9,)}}
    G = chain([("S", None)], faces)
    for node, port in ((1, 1), (2, 2)):  # a cap on each end
        G.add_node(node, unit=("<cap>", None))
        G.add_edge(0, node, bonds=(((0, port), (node, 9)),))
    assert free_faces(G, faces) == []


def test_named_cap_link_face_is_free() -> None:
    faces = {"S": LINEAR, "ester": {"cap": (2,), "link": (5,)}}
    G = chain([("S", None), ("S", None)], faces)
    G.add_node(2, unit=("ester", None))
    G.add_edge(1, 2, bonds=(((1, 2), (2, 2)),))
    assert free_faces(G, faces) == [(0, "head"), (2, "link")]


def test_partly_bonded_face_raises() -> None:
    faces = {"L": LADDER}
    G = chain([("L", None), ("L", None)], faces)
    G.edges[0, 1]["bonds"] = (((0, 3), (1, 1)),)  # only one strand of the junction
    with pytest.raises(CompatibilityError, match="partly bonded"):
        free_faces(G, faces)


# Site patterns
COMPOSITE_FACES = {
    ("PS", "S"): LINEAR,
    ("PMMA", "M"): LINEAR,
    ("PMMA", "ester"): {"cap": (2,), "link": (5,)},
    ("hub", "unit"): {1: (1,), 2: (2,)},
    ("star", "arm", "S"): LINEAR,
}
COMPONENTS = {"PS", "PMMA", "hub", "star"}


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("link", {(("PMMA", "ester"), "link")}),  # a face name, on any unit
        (5, {(("PMMA", "ester"), "link")}),  # a port label, read as the face holding it
        (("PS", "tail"), {(("PS", "S"), "tail")}),  # (component, x)
        (("PMMA", "M", "head"), {(("PMMA", "M"), "head")}),  # (component, unit, x)
        ((None, "S", "head"), {(("PS", "S"), "head"), (("star", "arm", "S"), "head")}),  # unit in any component
        (("star", ("arm", "S"), "tail"), {(("star", "arm", "S"), "tail")}),  # the rest of a path, exactly
        (("hub", 2), {(("hub", "unit"), 2)}),  # face names beat port labels (here they coincide)
        (("PMMA", None), {(("PMMA", "M"), "head"), (("PMMA", "M"), "tail"), (("PMMA", "ester"), "cap"), (("PMMA", "ester"), "link")}),
    ],
)
def test_composite_patterns(pattern, expected) -> None:
    assert resolve_pattern(pattern, COMPOSITE_FACES, COMPONENTS) == expected


def test_face_name_beats_port_label() -> None:
    faces = {"X": {"head": (1,), 1: (2,)}}  # port 1 is in 'head', but a face is NAMED 1
    assert resolve_pattern(1, faces) == {("X", 1)}


def test_any_matches_every_site() -> None:
    assert len(resolve_pattern(ANY, COMPOSITE_FACES, COMPONENTS)) == 2 + 2 + 2 + 2 + 2


def test_patterns_inside_one_component() -> None:
    faces = {"S": LINEAR, "M": LINEAR}
    assert resolve_pattern(("M", 2), faces) == {("M", "tail")}  # (unit, x)
    assert resolve_pattern("head", faces) == {("S", "head"), ("M", "head")}


@pytest.mark.parametrize(
    "pattern, components, message",
    [
        (("S", "head"), COMPONENTS, "isn't one of"),  # a unit label where a component belongs
        (("PS", "nope"), COMPONENTS, "matches no face"),
        ((1, 2, 3, 4), COMPONENTS, "Can't read"),
        (("S", "M", "head"), None, "Can't read"),  # three elements only at composite level
    ],
)
def test_pattern_errors(pattern, components, message) -> None:
    with pytest.raises(CompatibilityError, match=message):
        resolve_pattern(pattern, COMPOSITE_FACES if components else {"S": LINEAR}, components)


# Rules
def test_rules_are_directional_unless_symmetric() -> None:
    rule = head_to_tail()
    rule.validate(COMPOSITE_FACES, COMPONENTS)
    assert rule.compatible((("PS", "S"), "tail"), (("PMMA", "M"), "head"))
    assert not rule.compatible((("PMMA", "M"), "head"), (("PS", "S"), "tail"))

    symmetric = head_to_tail(symmetric=True)
    symmetric.validate(COMPOSITE_FACES, COMPONENTS)
    assert symmetric.compatible((("PMMA", "M"), "head"), (("PS", "S"), "tail"))


def test_only_equal_widths_are_compatible() -> None:
    faces = {("A", "S"): LINEAR, ("B", "L"): LADDER}
    rule = CompatibleSites(((None, None, "tail"), (None, None, "head")))
    rule.validate({**faces, ("C", "S"): LINEAR}, {"A", "B", "C"})
    assert not rule.compatible((("A", "S"), "tail"), (("B", "L"), "head"))

    with pytest.raises(CompatibilityError, match="equal width"):
        CompatibleSites((("A", "tail"), ("B", "head"))).validate(faces, {"A", "B"})


def test_rules_must_be_validated() -> None:
    with pytest.raises(RuntimeError, match="validated"):
        head_to_tail().compatible(("S", "tail"), ("S", "head"))


def test_candidates() -> None:
    rule = CompatibleSites((("hub", 1), (None, None, "head")))
    rule.validate(COMPOSITE_FACES, COMPONENTS)
    sites = [(("PS", "S"), "head"), (("PS", "S"), "tail"), (("PMMA", "M"), "head")]
    assert rule.candidates((("hub", "unit"), 1), sites) == [(("PS", "S"), "head"), (("PMMA", "M"), "head")]


# Matching
BLOCK_FACES = {("A", "S"): LINEAR, ("B", "M"): LINEAR}


def blocks() -> tuple[nx.Graph, nx.Graph]:
    return chain([(("A", "S"), None)] * 3, BLOCK_FACES), chain([(("B", "M"), None)] * 2, BLOCK_FACES)


def test_inferred_match_between_blocks() -> None:
    rule = head_to_tail()
    rule.validate(BLOCK_FACES, {"A", "B"})
    G_a, G_b = blocks()
    match = rule.match(G_a, free_faces(G_a, BLOCK_FACES), G_b, free_faces(G_b, BLOCK_FACES), np.random.default_rng(0))
    assert match == ((2, "tail"), (0, "head"))


def test_anchored_match() -> None:
    rule = CompatibleSites((ANY, (None, None, "head")))
    rule.validate(BLOCK_FACES, {"A", "B"})
    G_a, G_b = blocks()
    assert rule.match(G_a, [(0, "head")], G_b, free_faces(G_b, BLOCK_FACES), None) == ((0, "head"), (0, "head"))


def test_no_compatible_pair_raises() -> None:
    rule = CompatibleSites((("A", "head"), ("B", "head")))
    rule.validate(BLOCK_FACES, {"A", "B"})
    G_a, G_b = blocks()
    with pytest.raises(CompatibilityError, match="No compatible faces"):
        rule.match(G_a, [(2, "tail")], G_b, free_faces(G_b, BLOCK_FACES), None)


@pytest.mark.parametrize("if_ambiguous", ["error", "first", "random"])
def test_ambiguous_matches(if_ambiguous) -> None:
    rule = CompatibleSites((("A", "tail"), (None, None, None)), if_ambiguous=if_ambiguous)  # tail -> any face
    rule.validate(BLOCK_FACES, {"A", "B"})
    G_a, G_b = blocks()
    free_a, free_b = free_faces(G_a, BLOCK_FACES), free_faces(G_b, BLOCK_FACES)

    if if_ambiguous == "error":
        with pytest.raises(CompatibilityError, match="2 pairs"):
            rule.match(G_a, free_a, G_b, free_b, np.random.default_rng(0))
    elif if_ambiguous == "first":
        assert rule.match(G_a, free_a, G_b, free_b, None) == ((2, "tail"), (0, "head"))
    else:
        picks = {rule.match(G_a, free_a, G_b, free_b, np.random.default_rng(seed))[1] for seed in range(20)}
        assert picks == {(0, "head"), (1, "tail")}


def test_random_draws_once_per_match_even_without_a_tie() -> None:
    rule = head_to_tail(if_ambiguous="random")
    rule.validate(BLOCK_FACES, {"A", "B"})
    G_a, G_b = blocks()

    rng, reference = np.random.default_rng(3), np.random.default_rng(3)
    rule.match(G_a, free_faces(G_a, BLOCK_FACES), G_b, free_faces(G_b, BLOCK_FACES), rng)
    reference.random()
    assert rng.random() == reference.random()


def test_default_rule_draws_nothing() -> None:
    rule = head_to_tail()
    rule.validate(BLOCK_FACES, {"A", "B"})
    G_a, G_b = blocks()
    rule.match(G_a, free_faces(G_a, BLOCK_FACES), G_b, free_faces(G_b, BLOCK_FACES), None)  # None would fail if drawn from


def test_if_ambiguous_is_checked() -> None:
    with pytest.raises(ValueError, match="if_ambiguous"):
        head_to_tail(if_ambiguous="sometimes")
