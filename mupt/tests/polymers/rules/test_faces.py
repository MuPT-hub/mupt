"""Unit tests for face specs and port lookup on repeat units"""

import pytest
from rdkit import Chem

from mupt.chemistry.rdkit.sanitization import sanitized_mol
from mupt.chemistry.smiles import DEFAULT_SMILES_READ_PARAMS
from mupt.interfaces.smiles import primitive_from_smiles
from mupt.polymers.exceptions import FaceError
from mupt.polymers.rules.faces import (
    LINEAR_FACES,
    face_anchor_indices,
    free_ports_by_label,
    normalize_faces,
    port_atom_indices,
    strip_port_map_numbers,
    validate_faces,
)

# test units: ports are labelled by the map number on the "*" or on its anchor atom
LINEAR_SMILES = "*[CH2:1][CH:2](C)*"  # polypropylene, ports 1 (head) and 2 (tail)
LADDER_SMILES = "[*:1]c1cc([*:3])c([*:4])cc1[*:2]"  # benzene with four ports
LADDER_FACES = {"head": (1, 2), "tail": (3, 4)}


def unit(smiles: str):
    """Coordinate-free Primitive for a unit SMILES (faster than embedding)"""
    return primitive_from_smiles(smiles, embed_positions=False)


def mol_from(smiles: str) -> Chem.Mol:
    """H-explicit RDKit Mol, read the same way as primitive_from_smiles"""
    return sanitized_mol(Chem.MolFromSmiles(smiles, DEFAULT_SMILES_READ_PARAMS), add_Hs=True)


# normalize_faces
def test_normalize_none_gives_linear_convention() -> None:
    faces = normalize_faces(None)
    assert faces == {"head": (1,), "tail": (2,)} == dict(LINEAR_FACES)

    # the returned dict is a copy; editing it mustn't change the shared default
    faces["head"] = (9,)
    assert LINEAR_FACES["head"] == (1,)


def test_normalize_coerces_values_to_tuples() -> None:
    assert normalize_faces({"head": 1, "tail": [2]}) == {"head": (1,), "tail": (2,)}
    assert normalize_faces(LADDER_FACES) == LADDER_FACES
    assert normalize_faces({"a": 1, "b1": 2, "b2": 3}) == {"a": (1,), "b1": (2,), "b2": (3,)}


@pytest.mark.parametrize(
    "raw",
    [
        {"head": (), "tail": (2,)},  # empty face
        {"head": (1, 1), "tail": (2,)},  # port repeated within a face
        {"head": (1,), "tail": (1,)},  # port shared between faces
        [(1,), (2,)],  # not a mapping
    ],
)
def test_normalize_rejects_bad_specs(raw) -> None:
    with pytest.raises(FaceError):
        normalize_faces(raw)


# free_ports_by_label
def test_free_ports_ignore_internal_connectors() -> None:
    # the CH2 -> CH bond also carries map-number labels, but only free ports count
    prim = unit("[*:7][CH2:1][CH:2](C)*")
    assert set(free_ports_by_label(prim)) == {7, 2}


def test_free_ports_reject_duplicate_labels() -> None:
    with pytest.raises(FaceError):
        free_ports_by_label(unit("[*:1]C[*:1]"))


# validate_faces
def test_validate_accepts_matching_specs() -> None:
    validate_faces(unit(LINEAR_SMILES), LINEAR_FACES, required=("head", "tail"))
    validate_faces(unit(LADDER_SMILES), LADDER_FACES, required=("head", "tail"))


@pytest.mark.parametrize(
    "smiles, faces, required",
    [
        (LINEAR_SMILES, {"head": (1,), "tail": (5,)}, ()),  # port 5 doesn't exist
        (LADDER_SMILES, {"head": (1, 2), "tail": (3,)}, ()),  # free port 4 isn't in a face
        (LINEAR_SMILES, {"a": (1,), "b": (2,)}, ("head", "tail")),  # required faces missing
        ("[*:1]=CC[*:2]", {"side": (1, 2)}, ()),  # double and single bond in one face
    ],
)
def test_validate_rejects_mismatched_specs(smiles, faces, required) -> None:
    with pytest.raises(FaceError):
        validate_faces(unit(smiles), faces, required=required)


def test_validate_names_the_unit_in_errors() -> None:
    with pytest.raises(FaceError, match="styrene"):
        validate_faces(unit(LINEAR_SMILES), {"head": (1,)}, label="styrene")


# strip_port_map_numbers
def test_strip_only_touches_port_anchors() -> None:
    prim = unit("*[CH2:1][CH2:5][CH2:2]*")
    strip_port_map_numbers(prim, LINEAR_FACES)

    remaining = {atom.metadata.get("molAtomMapNumber") for atom in prim.children}
    assert 5 in remaining  # a non-port annotation survives
    assert not ({1, 2} & remaining)  # both port anchors were stripped


def test_ports_are_still_found_after_stripping() -> None:
    # labels are set on import, so removing the metadata doesn't change port lookup
    prim = unit(LINEAR_SMILES)
    strip_port_map_numbers(prim, LINEAR_FACES)
    assert set(free_ports_by_label(prim)) == {1, 2}


# RDKit-level helpers
def test_port_atom_indices_use_importer_labelling() -> None:
    mol = mol_from("[*:7][CH2:1][CH:2](C)*")
    ports = port_atom_indices(mol)
    assert set(ports) == {7, 2}  # "*" map number wins over the anchor's

    anchor, linker = ports[2]
    assert mol.GetAtomWithIdx(anchor).GetAtomMapNum() == 2
    assert mol.GetAtomWithIdx(linker).GetAtomicNum() == 0


def test_face_anchor_indices_follow_strand_order() -> None:
    mol = mol_from(LADDER_SMILES)
    ports = port_atom_indices(mol)

    assert face_anchor_indices(mol, LADDER_FACES, "tail") == (ports[3][0], ports[4][0])
    assert face_anchor_indices(mol, {"tail": (4, 3)}, "tail") == (ports[4][0], ports[3][0])


@pytest.mark.parametrize("faces, name", [(LADDER_FACES, "side"), ({"head": (1, 9)}, "head")])
def test_face_anchor_indices_reject_unknown_faces_and_ports(faces, name) -> None:
    with pytest.raises(FaceError):
        face_anchor_indices(mol_from(LADDER_SMILES), faces, name)
