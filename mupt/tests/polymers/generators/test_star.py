"""Unit tests for StarPolymerGenerator"""

import pytest

from mupt.roles import has_SAAMR_roles
from mupt.polymers.exceptions import CompatibilityError
from mupt.polymers.generators.linear import LinearPolymerGenerator
from mupt.polymers.generators.star import StarPolymerGenerator
from mupt.polymers.rules.compatibility import ANY, CompatibleSites

HUB = "[*:1]C([*:2])([*:3])C([*:4])([*:5])[*:6]"  # six ports, one face each
STYRENE = "*[CH2:1][CH:2](c1ccccc1)*"
PE = "*[CH2:1][CH2:2]*"


def unit_keys(G) -> list:
    return [G.nodes[node]["unit"] for node in G]


def arm(units, size, **kwargs) -> LinearPolymerGenerator:
    return LinearPolymerGenerator(units, size=size, end_groups=kwargs.pop("end_groups", {"tail": "[H]*"}), **kwargs)


def two_arm_types(**kwargs) -> StarPolymerGenerator:
    return StarPolymerGenerator(
        {"hub": HUB, "arm_A": arm({"S": STYRENE}, 2), "arm_B": arm({"PE": PE}, 3)},
        core="hub",
        arms={(1, 2, 3): "arm_A", (4, 5, 6): "arm_B"},
        **kwargs,
    )


# Building stars
def test_six_arms_on_the_right_ports() -> None:
    G = two_arm_types().generate_topologies(1, seed=0)[0]

    assert G.number_of_nodes() == 1 + 3 * (2 + 1) + 3 * (3 + 1)
    assert G.nodes[0]["unit"] == (("hub", "unit"), None)

    # each core port bonds to the head (port 1) of the first unit of its arm
    arm_on_port = {}
    for _, other, bonds in G.edges(0, data="bonds"):
        ((core, core_port), (arm_node, arm_port)) = bonds[0]
        assert (core, arm_port) == (0, 1)
        arm_on_port[core_port] = G.nodes[arm_node]["unit"][0][0]
    assert arm_on_port == {1: "arm_A", 2: "arm_A", 3: "arm_A", 4: "arm_B", 5: "arm_B", 6: "arm_B"}


def test_capped_star_realizes_closed() -> None:
    universe = two_arm_types().realize(2, seed=0)
    assert all(len(star.connections.connectors_free) == 0 for star in universe.children)
    assert has_SAAMR_roles(universe)


def test_arms_attach_through_a_named_cap_link() -> None:
    # arms whose head is H-capped, exposing only an ester linker's port 5 at the tail
    ester_arm = LinearPolymerGenerator(
        {"S": STYRENE},
        size=2,
        end_groups={"head": "[H]*", "tail": {"ester": "[*:5]C(=O)[*:2]"}},
        faces={"ester": {"cap": 2, "link": 5}},
    )
    star = StarPolymerGenerator(
        {"hub": "[*:1]C([*:2])([*:3])[*:4]", "arm": ester_arm},
        core="hub",
        arms={(1, 2, 3, 4): "arm"},
        compatibility=CompatibleSites((("hub", None), ("arm", "link"))),
    )
    G = star.generate_topologies(1, seed=0)[0]

    for _, other, bonds in G.edges(0, data="bonds"):
        assert G.nodes[other]["unit"] == (("arm", "ester"), None)
        assert bonds[0][1][1] == 5  # bonded through the ester's link port
    assert len(star.realize(1, seed=0).children[0].connections.connectors_free) == 0


def test_generated_core_gets_one_arm_per_matching_face() -> None:
    # a capped 3-unit backbone whose units each carry a side port 9: a comb-like core
    backbone = LinearPolymerGenerator(
        {"X": "*[CH2:1][CH:2]([*:9])*"},
        size=3,
        faces={"X": {"head": 1, "tail": 2, "side": 9}},
        end_groups="[H]*",
    )
    star = StarPolymerGenerator({"core": backbone, "arm": arm({"PE": PE}, 2)}, core="core", arms={"side": "arm"})
    G = star.generate_topologies(1, seed=0)[0]

    labels = [key[0] for key in unit_keys(G)]
    assert labels.count(("arm", "PE")) == 3 * 2
    assert len(star.realize(1, seed=0).children[0].connections.connectors_free) == 0


# Free core faces
def test_unfilled_core_faces_raise_unless_allowed() -> None:
    components = {"hub": HUB, "arm": arm({"PE": PE}, 2)}
    with pytest.raises(CompatibilityError, match="have no arm"):
        StarPolymerGenerator(components, core="hub", arms={(1, 2, 3): "arm"})

    star = StarPolymerGenerator(components, core="hub", arms={(1, 2, 3): "arm"}, allow_free_core_ports=True)
    chain = star.realize(1, seed=0).children[0]
    assert sorted(conn.label for conn in chain.connections.connectors_free) == [4, 5, 6]


# Ambiguous arm faces
@pytest.mark.parametrize("if_ambiguous", ["error", "random"])
def test_uncapped_arms_need_a_choice(if_ambiguous) -> None:
    star = StarPolymerGenerator(
        {"hub": "[*:1]C[*:2]", "arm": LinearPolymerGenerator({"PE": PE}, size=2)},  # both arm ends free
        core="hub",
        arms={(1, 2): "arm"},
        compatibility=CompatibleSites((ANY, (None, None, None)), if_ambiguous=if_ambiguous),
    )
    if if_ambiguous == "error":
        with pytest.raises(CompatibilityError, match="pairs of faces fit"):
            star.generate_topologies(1, seed=0)
    else:
        assert [unit_keys(G) for G in star.generate_topologies(3, seed=4)] == [
            unit_keys(G) for G in star.generate_topologies(3, seed=4)
        ]


# Errors
@pytest.mark.filterwarnings("ignore:Component")  # each case leaves one of the components unused
@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(arms={(1, 2): "arm", (2, 3, 4, 5, 6): "arm"}), CompatibilityError, "several arms keys"),
        (dict(arms={7: "arm"}), CompatibilityError, "matches no face"),
        (dict(arms={(1, 2, 3, 4, 5, 6): "nope"}), ValueError, "aren't among"),
        (dict(core="nope", arms={1: "arm"}), ValueError, "Core 'nope'"),
        (dict(arms={}), ValueError, "At least one arm"),
        (dict(arms={(1, 2, 3, 4, 5, 6): "capped"}), CompatibilityError, "no free site compatible"),
    ],
)
def test_constructor_errors(kwargs, error, message) -> None:
    components = {"hub": HUB, "arm": arm({"PE": PE}, 2), "capped": arm({"PE": PE}, 2, end_groups="[H]*")}
    kwargs.setdefault("core", "hub")
    with pytest.raises(error, match=message):
        StarPolymerGenerator(components, **kwargs)
