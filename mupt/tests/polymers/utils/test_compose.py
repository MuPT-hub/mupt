"""Unit tests for composition: namespacing, resolving, and flattening component graphs"""

import networkx as nx
import numpy as np
import pytest

from mupt.polymers._common import FixedFace
from mupt.polymers.exceptions import CompatibilityError, TopologyValidationError
from mupt.polymers.generators.linear import LinearPolymerGenerator
from mupt.polymers.models import isotactic
from mupt.polymers.rules.compatibility import head_to_tail
from mupt.polymers.rules.junctions import FixedPairing
from mupt.polymers.utils.compose import (
    flatten_component_graph,
    namespace_faces,
    namespace_key,
    namespace_lexicon,
    resolve_component_graph,
    validate_component_graph,
)
from mupt.polymers.utils.realize import validate_descriptive_graph

PE = "*[CH2:1][CH2:2]*"
PP = "*[CH2:1][CH:2](C)*"


# Namespacing
def test_keys_are_namespaced_as_flat_paths() -> None:
    assert namespace_key("PS", ("S", 1)) == (("PS", "S"), 1)
    assert namespace_key("star", (("PS", "S"), 1)) == (("star", "PS", "S"), 1)  # stays flat when nested
    assert namespace_key("A", ("<tail_cap>", None)) == (("A", "<tail_cap>"), None)


def test_only_generated_keys_can_be_namespaced() -> None:
    with pytest.raises(TypeError):
        namespace_key("A", "PE")


def test_namespaced_templates_are_copies_recording_their_components() -> None:
    lexicon = LinearPolymerGenerator({"PP": PP}, size=2, tacticity=isotactic()).lexicon
    inner = namespace_lexicon("B", lexicon)
    outer = namespace_lexicon("arm", inner)

    template = outer[(("arm", "B", "PP"), 1)]
    assert template.metadata["component"] == ("arm", "B")
    assert template.metadata["unit_label"] == "PP"
    assert template is not inner[(("B", "PP"), 1)]
    assert "component" not in lexicon[("PP", 1)].metadata  # the original is untouched


def test_faces_are_namespaced() -> None:
    assert namespace_faces("A", {"S": {"head": (1,)}}) == {("A", "S"): {"head": (1,)}}


# Resolving and flattening
GENERATORS = {
    "A": LinearPolymerGenerator({"PE": PE}, size=3),
    "B": LinearPolymerGenerator({"PP": PP}, size=2, tacticity=isotactic()),
}
FACES = {**namespace_faces("A", GENERATORS["A"].faces), **namespace_faces("B", GENERATORS["B"].faces)}
LEXICON = {**namespace_lexicon("A", GENERATORS["A"].lexicon), **namespace_lexicon("B", GENERATORS["B"].lexicon)}


def coarse_path(labels, anchors=None) -> nx.Graph:
    """An unresolved path of components, joined with the given (or no) anchors"""
    G = nx.path_graph(len(labels))
    nx.set_node_attributes(G, dict(enumerate(labels)), "component")
    for i in range(len(labels) - 1):
        G.edges[i, i + 1]["joins"] = ((i, None), (i + 1, None)) if anchors is None else anchors[i]
    return G


def resolve(coarse, rng=0, compatibility=None) -> nx.Graph:
    rule = compatibility or head_to_tail()
    rule.validate(FACES, set(GENERATORS))
    return resolve_component_graph(coarse, GENERATORS, FACES, rule, FixedPairing(), np.random.default_rng(rng))


def test_resolving_adds_topologies_and_addressed_bonds() -> None:
    coarse = coarse_path(["A", "B"])
    resolved = resolve(coarse)

    assert "topology" not in coarse.nodes[0]  # the input is left unresolved
    assert [resolved.nodes[0]["topology"].nodes[n]["unit"] for n in range(3)] == [(("A", "PE"), None)] * 3
    assert resolved.edges[0, 1]["bonds"] == ((((0, 2), 2), ((1, 0), 1)),)  # A's last tail -> B's first head
    assert resolved.edges[0, 1]["joins"] == ((0, None), (1, None))


def test_components_draw_in_node_order() -> None:
    resolved = resolve(coarse_path(["A", "B"]), rng=7)

    rng = np.random.default_rng(7)
    GENERATORS["A"].generate_topology(rng)
    expected_b = GENERATORS["B"].generate_topology(rng)
    assert [key[1] for _, key in resolved.nodes[1]["topology"].nodes(data="unit")] == [
        key[1] for _, key in expected_b.nodes(data="unit")
    ]


def test_a_free_face_joins_only_once() -> None:
    # A-B-A: B's head is used by the first join, so the second must use its tail
    resolved = resolve(coarse_path(["A", "B", "A"]))
    assert resolved.edges[1, 2]["bonds"] == ((((1, 1), 2), ((2, 0), 1)),)


def test_flattened_graph_is_a_valid_unit_graph() -> None:
    G = flatten_component_graph(resolve(coarse_path(["A", "B", "A"])))

    assert list(G) == list(range(3 + 2 + 3))
    assert G.edges[2, 3]["bonds"] == (((2, 2), (3, 1)),)  # the A-B join, renumbered
    assert G.edges[4, 5]["bonds"] == (((4, 2), (5, 1)),)  # the B-A join
    validate_descriptive_graph(G, LEXICON)


def test_flattening_needs_a_resolved_graph() -> None:
    with pytest.raises(TopologyValidationError, match="isn't resolved"):
        flatten_component_graph(coarse_path(["A", "B"]))


def test_fixed_face_and_pattern_anchors() -> None:
    # bond A's HEAD (a fixed face) to B's tail (named by a pattern), against the usual direction
    resolved = resolve(coarse_path(["A", "B"]))
    coarse = coarse_path(["A", "B"], anchors=[((0, FixedFace(0, "head")), (1, "tail"))])
    coarse.nodes[0]["topology"] = resolved.nodes[0]["topology"]  # resolved early, as a star does

    out = resolve(coarse, compatibility=head_to_tail(symmetric=True))
    assert out.edges[0, 1]["bonds"] == ((((0, 0), 1), ((1, 1), 2)),)


def test_fixed_face_must_be_free() -> None:
    resolved = resolve(coarse_path(["A", "B"]))
    coarse = coarse_path(["A", "B"], anchors=[((0, FixedFace(1, "head")), (1, None))])  # node 1's head is bonded
    coarse.nodes[0]["topology"] = resolved.nodes[0]["topology"]
    with pytest.raises(CompatibilityError, match="isn't free"):
        resolve(coarse)


def test_duplicate_join_edges_raise() -> None:
    resolved = resolve(coarse_path(["A", "B"]))
    resolved.edges[0, 1]["bonds"] = ((((0, 0), 2), ((0, 1), 1)),)  # a hand edit landing on an edge inside A
    with pytest.raises(TopologyValidationError, match="already share an edge"):
        flatten_component_graph(resolved)


def test_validation_of_component_graphs() -> None:
    validate_component_graph(coarse_path(["A", "B"]), GENERATORS)
    with pytest.raises(TopologyValidationError, match="unknown component"):
        validate_component_graph(coarse_path(["A", "C"]), GENERATORS)
    with pytest.raises(TopologyValidationError, match="don't match"):
        validate_component_graph(coarse_path(["A", "B"], anchors=[((0, None), (5, None))]), GENERATORS)
