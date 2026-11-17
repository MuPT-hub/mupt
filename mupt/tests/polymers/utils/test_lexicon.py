"""Unit tests for building lexicon templates of repeat units and chain-end caps"""

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from mupt.interfaces.rdkit import rdkit_mol_from_primitive
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.polymers.exceptions import FaceError
from mupt.polymers.rules.faces import LINEAR_FACES, free_ports_by_label
from mupt.polymers.utils.lexicon import (
    CAP_FACE,
    build_cap_template,
    build_unit_templates,
    cap_key,
)
from mupt.polymers.utils.realize import primitive_from_descriptive_graph

PP = "*[CH2:1][CH:2](C)*"  # polypropylene: one backbone stereocenter
PE = "*[CH2:1][CH2:2]*"  # polyethylene: none
TTSBI = "C12(CC(C)(C)c3cc(O[*:3])c(O[*:4])cc31)CC(C)(C)c1cc(O[*:1])c(O[*:2])cc12"
TTSBI_FACES = {"head": (1, 2), "tail": (3, 4)}


def cip_labels(template) -> list[str]:
    """CIP labels (R/S) of every stereocenter in a template, read back through RDKit"""
    mol = rdkit_mol_from_primitive(template)
    return [code for _, code in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)]


def port_map_numbers(template) -> set:
    """Map numbers still stored on the template's atoms"""
    return {atom.metadata.get("molAtomMapNumber") for atom in template.children} - {None}


# Which variants get built
def test_tactic_unit_gets_two_variants() -> None:
    lexicon = build_unit_templates("PP", PP, LINEAR_FACES, tactic=True)
    assert set(lexicon) == {("PP", 1), ("PP", -1)}


def test_untactic_or_achiral_units_get_one_template() -> None:
    assert set(build_unit_templates("PP", PP, LINEAR_FACES)) == {("PP", None)}  # tacticity off
    assert set(build_unit_templates("PE", PE, LINEAR_FACES, tactic=True)) == {("PE", None)}  # no stereocenter


def test_variants_have_opposite_handedness() -> None:
    lexicon = build_unit_templates("PP", PP, LINEAR_FACES, tactic=True)
    plus, minus = cip_labels(lexicon[("PP", 1)]), cip_labels(lexicon[("PP", -1)])

    assert len(plus) == len(minus) == 1
    assert {plus[0], minus[0]} == {"R", "S"}


def test_ladder_unit_varies_its_spiro_center() -> None:
    lexicon = build_unit_templates("TTSBI", TTSBI, TTSBI_FACES, tactic=True)
    assert set(lexicon) == {("TTSBI", 1), ("TTSBI", -1)}


# Finishing templates
def test_templates_carry_residue_metadata() -> None:
    lexicon = build_unit_templates("PP", PP, LINEAR_FACES, tactic=True)

    for (label, config), template in lexicon.items():
        assert template.metadata["unit_label"] == label
        assert template.metadata["config"] == config
        assert template.metadata["residue_name"] == "PP"

        # copies are what get realized, so they must inherit the metadata
        assert template.copy().metadata == template.metadata


def test_port_map_numbers_are_stripped_but_ports_still_found() -> None:
    template = build_unit_templates("PE", "*[CH2:1][CH2:5][CH2:2]*", LINEAR_FACES)[("PE", None)]

    assert port_map_numbers(template) == {5}  # only the non-port annotation is left
    assert set(free_ports_by_label(template)) == {1, 2}


# Primitive-valued units
def test_primitive_unit_is_copied_with_its_geometry() -> None:
    unit = primitive_from_smiles(PE, label="PE", embed_random_seed=0)
    template = build_unit_templates("PE", unit, LINEAR_FACES)[("PE", None)]

    assert template is not unit
    assert port_map_numbers(unit) == {1, 2}  # the user's Primitive is left untouched

    # no re-embedding: every atom sits exactly where the user put it
    for given, copied in zip(unit.children, template.children):
        assert np.array_equal(given.shape.positions, copied.shape.positions)


def test_tactic_primitive_unit_gets_two_variants() -> None:
    unit = primitive_from_smiles(PP, label="PP", embed_positions=False)
    assert set(build_unit_templates("PP", unit, LINEAR_FACES, tactic=True)) == {("PP", 1), ("PP", -1)}


# Errors
def test_bad_faces_raise() -> None:
    with pytest.raises(FaceError):
        build_unit_templates("PE", PE, {"head": (1,), "tail": (7,)})


def test_required_faces_are_checked() -> None:
    with pytest.raises(FaceError, match="required"):
        build_unit_templates("PE", PE, {"a": (1,), "b": (2,)}, required_faces=("head", "tail"))


def test_tactic_units_need_head_and_tail_faces() -> None:
    with pytest.raises(FaceError, match="'head' and 'tail'"):
        build_unit_templates("PP", PP, {"a": (1,), "b": (2,)}, tactic=True)


# Caps
def test_single_port_cap() -> None:
    template, faces = build_cap_template("[H]*", "head")

    assert list(faces) == [CAP_FACE]
    assert len(faces[CAP_FACE]) == 1
    assert template.metadata == {"unit_label": "head_cap", "config": None, "residue_name": "head_cap"}


def test_multi_port_cap_orders_ports_by_label() -> None:
    _, faces = build_cap_template("[*:2]OCO[*:1]", "tail")
    assert faces == {CAP_FACE: (1, 2)}


def test_cap_keys_are_reserved_per_end() -> None:
    assert cap_key("head") == ("<head_cap>", None)
    assert cap_key("tail") == ("<tail_cap>", None)


@pytest.mark.parametrize("cap, side, error", [("[H]*", "middle", ValueError), ("C", "head", FaceError)])
def test_bad_caps_raise(cap, side, error) -> None:
    with pytest.raises(error):
        build_cap_template(cap, side)


# Templates feed straight into realization
def test_templates_realize_into_a_capped_chain() -> None:
    lexicon = build_unit_templates("PP", PP, LINEAR_FACES, tactic=True)
    head_cap, head_faces = build_cap_template("[H]*", "head")
    tail_cap, tail_faces = build_cap_template("[H]*", "tail")
    lexicon |= {cap_key("head"): head_cap, cap_key("tail"): tail_cap}
    (head_port,), (tail_port,) = head_faces[CAP_FACE], tail_faces[CAP_FACE]

    # head cap - PP(+1) - PP(-1) - tail cap
    G = nx.path_graph(4)
    for node, key in enumerate([cap_key("head"), ("PP", 1), ("PP", -1), cap_key("tail")]):
        G.nodes[node]["unit"] = key
    G.edges[0, 1]["bonds"] = (((0, head_port), (1, 1)),)
    G.edges[1, 2]["bonds"] = (((1, 2), (2, 1)),)
    G.edges[2, 3]["bonds"] = (((2, 2), (3, tail_port)),)

    chain = primitive_from_descriptive_graph(G, lexicon)
    assert len(chain.connections.connectors_free) == 0
    assert [unit.metadata["config"] for unit in chain.children] == [None, 1, -1, None]


# Choosing the side group that defines configs
MMA_LABELLED = "*[CH2:1][C:2]([CH3:5])(C(=O)OC)*"  # methyl labelled 5, for overriding


def test_side_group_override_swaps_the_variants() -> None:
    default = build_unit_templates("MMA", MMA_LABELLED, LINEAR_FACES, tactic=True)
    override = build_unit_templates("MMA", MMA_LABELLED, LINEAR_FACES, tactic=True, side_group=5)

    # X = methyl instead of the ester: the +1 template becomes the default's -1 template
    assert cip_labels(override[("MMA", 1)]) == cip_labels(default[("MMA", -1)])
    assert cip_labels(override[("MMA", 1)]) != cip_labels(default[("MMA", 1)])


def test_side_group_map_number_is_stripped() -> None:
    for template in build_unit_templates("MMA", MMA_LABELLED, LINEAR_FACES, tactic=True, side_group=5).values():
        assert 5 not in port_map_numbers(template)


@pytest.mark.parametrize(
    "side_group, message",
    [(9, "no atom with map number 9"), (2, "also a port label")],
)
def test_bad_side_groups_raise(side_group, message) -> None:
    with pytest.raises(FaceError, match=message):
        build_unit_templates("MMA", MMA_LABELLED, LINEAR_FACES, tactic=True, side_group=side_group)
