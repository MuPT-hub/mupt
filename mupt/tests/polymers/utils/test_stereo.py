"""Unit tests for backbone stereocenters and backbone-relative stereo variants of repeat units"""

import pytest
from rdkit import Chem

from mupt.chemistry.rdkit.sanitization import sanitized_mol
from mupt.chemistry.smiles import DEFAULT_SMILES_READ_PARAMS
from mupt.polymers.exceptions import FaceError
from mupt.polymers.rules.faces import LINEAR_FACES, port_atom_indices
from mupt.polymers.utils.stereo import (
    backbone_atom_indices,
    backbone_stereocenters,
    canonical_neighbors,
    stereo_variant,
    strand_paths,
)

TTSBI = "C12(CC(C)(C)c3cc(O[*:3])c(O[*:4])cc31)CC(C)(C)c1cc(O[*:1])c(O[*:2])cc12"
TTSBI_FACES = {"head": (1, 2), "tail": (3, 4)}

# the same monomers, written several ways (atom order changes; structure doesn't)
STYRENE_SPELLINGS = [
    "*[CH2:1][CH:2](c1ccccc1)*",
    "*[CH2:1][CH:2](*)c1ccccc1",
    "c1ccccc1[CH:2](*)[CH2:1]*",
    "*[CH:2](c1ccccc1)[CH2:1]*",
]
MMA_SPELLINGS = [
    "*[CH2:1][C:2](C)(C(=O)OC)*",
    "*[CH2:1][C:2](C(=O)OC)(C)*",
    "COC(=O)[C:2](C)(*)[CH2:1]*",
    "*[C:2](C)(C(=O)OC)[CH2:1]*",
]


def mol_from(smiles: str) -> Chem.Mol:
    """H-explicit RDKit Mol, read the same way as primitive_from_smiles"""
    return sanitized_mol(Chem.MolFromSmiles(smiles, DEFAULT_SMILES_READ_PARAMS), add_Hs=True)


def variant(smiles: str, config: int, faces=LINEAR_FACES, side_groups=()) -> Chem.Mol:
    """The given config's stereo variant of a unit"""
    mol = mol_from(smiles)
    paths = strand_paths(mol, faces)
    return stereo_variant(mol, backbone_stereocenters(mol, paths), config, paths, side_groups)


def structure_smiles(mol: Chem.Mol) -> str:
    """Canonical isomeric SMILES ignoring atom map numbers (i.e. port labels)"""
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


# Strand paths and the backbone
def test_strand_path_runs_linker_to_linker() -> None:
    mol = mol_from("*[CH2:1][CH:2](C)*")
    ports = port_atom_indices(mol)
    (path,) = strand_paths(mol, LINEAR_FACES)

    assert path[0] == ports[1][1] and path[-1] == ports[2][1]  # head and tail "*" linkers
    assert path[1:-1] == [ports[1][0], ports[2][0]]  # head anchor, then tail anchor


def test_single_atom_backbone() -> None:
    # both ports sit on one carbon, so the backbone is just that atom
    mol = mol_from("[*:1][CH](C)[*:2]")
    paths = strand_paths(mol, LINEAR_FACES)
    assert len(backbone_atom_indices(paths)) == 1


def test_ladder_backbone_unions_both_strands() -> None:
    # TTSBI's two strands meet only at the spiro carbon (atom 0), so it lies on the backbone
    mol = mol_from(TTSBI)
    paths = strand_paths(mol, TTSBI_FACES)
    assert len(paths) == 2
    assert all(0 in path for path in paths)


@pytest.mark.parametrize(
    "faces",
    [
        {"head": (1, 2), "tail": (3,)},  # one tail port missing a partner
        {"a": (1, 2), "b": (3, 4)},  # no head/tail faces
        {"head": (1, 2), "tail": (3, 9)},  # no port 9
    ],
)
def test_strand_paths_reject_bad_faces(faces) -> None:
    with pytest.raises(FaceError):
        strand_paths(mol_from(TTSBI), faces)


# Backbone stereocenters
def test_polypropylene_has_one_backbone_stereocenter() -> None:
    mol = mol_from("*[CH2:1][CH:2](C)*")
    centers = backbone_stereocenters(mol, strand_paths(mol, LINEAR_FACES))

    assert len(centers) == 1
    assert mol.GetAtomWithIdx(centers[0]).GetAtomMapNum() == 2  # the CH, not the CH2


def test_polyethylene_has_no_stereocenters() -> None:
    mol = mol_from("*[CH2:1][CH2:2]*")
    assert backbone_stereocenters(mol, strand_paths(mol, LINEAR_FACES)) == []


def test_side_chain_stereocenters_are_not_backbone() -> None:
    # a chiral sec-butyl ester pendant is a stereocenter, but not on the backbone
    mol = mol_from("*[CH2:1][CH:2](C(=O)O[C@@H](C)CC)*")
    centers = backbone_stereocenters(mol, strand_paths(mol, LINEAR_FACES))
    assert [mol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in centers] == [2]


def test_ttsbi_spiro_carbon_is_its_backbone_stereocenter() -> None:
    mol = mol_from(TTSBI)
    assert backbone_stereocenters(mol, strand_paths(mol, TTSBI_FACES)) == [0]


# Canonical neighbor order
def test_canonical_order_is_head_tail_then_side_groups() -> None:
    mol = mol_from("*[CH2:1][CH:2](c1ccccc1)*")
    ports = port_atom_indices(mol)
    (path,) = paths = strand_paths(mol, LINEAR_FACES)
    center = ports[2][0]

    head_side, tail_side, x, rest = canonical_neighbors(mol, center, paths)
    assert (head_side, tail_side) == (ports[1][0], ports[2][1])  # the CH2, then the tail "*"
    assert mol.GetAtomWithIdx(x).GetIsAromatic()  # phenyl outranks H
    assert mol.GetAtomWithIdx(rest).GetSymbol() == "H"


def test_default_side_group_of_mma_is_the_ester() -> None:
    mol = mol_from(MMA_SPELLINGS[0])
    paths = strand_paths(mol, LINEAR_FACES)
    (center,) = backbone_stereocenters(mol, paths)

    _, _, x, rest = canonical_neighbors(mol, center, paths)
    assert any(nbr.GetSymbol() == "O" for nbr in mol.GetAtomWithIdx(x).GetNeighbors())  # carbonyl C
    assert mol.GetAtomWithIdx(rest).GetDegree() == 4  # the methyl (C + 3 H)


def test_side_group_override() -> None:
    mol = mol_from("*[CH2:1][C:2]([CH3:5])(C(=O)OC)*")
    paths = strand_paths(mol, LINEAR_FACES)
    (center,) = backbone_stereocenters(mol, paths)
    methyl = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 5)

    assert canonical_neighbors(mol, center, paths, side_groups=[methyl])[2] == methyl


def test_spiro_center_is_ordered_by_its_strands() -> None:
    mol = mol_from(TTSBI)
    paths = strand_paths(mol, TTSBI_FACES)
    order = canonical_neighbors(mol, 0, paths)

    # both strands cross the spiro carbon through the same two aromatic carbons (ports 1, 2
    # share a catechol ring), so those come first; the two CH2 arms follow as side groups
    i = paths[0].index(0)
    assert order[:2] == (paths[0][i - 1], paths[0][i + 1])
    assert all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in order[:2])
    assert sorted(order) == sorted(nbr.GetIdx() for nbr in mol.GetAtomWithIdx(0).GetNeighbors())

    # the CH2 arms tie on CIP rank; the one in the head-side neighbor's five-membered ring comes first
    rings = mol.GetRingInfo().AtomRings()
    assert any({0, order[0], order[2]} <= set(ring) for ring in rings)
    assert not any({0, order[0], order[3]} <= set(ring) for ring in rings)


def test_ttsbi_variants_dont_depend_on_atom_order() -> None:
    # the same TTSBI with its atoms renumbered gives the same +1 variant
    mol = mol_from(TTSBI)
    renumbered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    results = set()
    for unit in (mol, renumbered):
        paths = strand_paths(unit, TTSBI_FACES)
        results.add(structure_smiles(stereo_variant(unit, backbone_stereocenters(unit, paths), 1, paths)))
    assert len(results) == 1


def test_map_numbers_dont_sway_the_default_side_group() -> None:
    # labelling MMA's methyl (e.g. to override with it later) mustn't make it the default X
    mol = mol_from("*[CH2:1][C:2]([CH3:5])(C(=O)OC)*")
    paths = strand_paths(mol, LINEAR_FACES)
    (center,) = backbone_stereocenters(mol, paths)

    x = canonical_neighbors(mol, center, paths)[2]
    assert any(nbr.GetSymbol() == "O" for nbr in mol.GetAtomWithIdx(x).GetNeighbors())  # still the ester


def test_tied_side_groups_raise() -> None:
    # a backbone carbon with two identical side groups has no side group to choose;
    # call canonical_neighbors directly, since such an atom isn't really a stereocenter
    mol = mol_from("*[CH2:1][C:2](C)(C)*")
    paths = strand_paths(mol, LINEAR_FACES)
    center = port_atom_indices(mol)[2][0]
    with pytest.raises(ValueError, match="rank equally"):
        canonical_neighbors(mol, center, paths)


# Stereo variants
@pytest.mark.parametrize("spellings", [STYRENE_SPELLINGS, MMA_SPELLINGS], ids=["styrene", "MMA"])
@pytest.mark.parametrize("config", [1, -1])
def test_variants_dont_depend_on_how_the_smiles_is_written(spellings, config) -> None:
    assert len({structure_smiles(variant(smiles, config)) for smiles in spellings}) == 1


@pytest.mark.parametrize("smiles", STYRENE_SPELLINGS[:1] + MMA_SPELLINGS[:1])
def test_variants_are_mirror_configurations(smiles) -> None:
    plus, minus = structure_smiles(variant(smiles, 1)), structure_smiles(variant(smiles, -1))
    assert plus != minus


def test_side_group_override_flips_the_configuration() -> None:
    # choosing the methyl instead of the ester as X swaps which variant is +1
    smiles = "*[CH2:1][C:2]([CH3:5])(C(=O)OC)*"
    methyl = next(atom.GetIdx() for atom in mol_from(smiles).GetAtoms() if atom.GetAtomMapNum() == 5)

    default_plus = structure_smiles(variant(smiles, 1))
    override_plus = structure_smiles(variant(smiles, 1, side_groups=[methyl]))
    assert override_plus == structure_smiles(variant(smiles, -1))
    assert override_plus != default_plus


def test_input_stereo_is_replaced() -> None:
    # stereo written in the SMILES doesn't define +1; both inputs give the same variants
    for config in (1, -1):
        given_r = structure_smiles(variant("*[CH2:1][C@H:2](C)*", config))
        given_s = structure_smiles(variant("*[CH2:1][C@@H:2](C)*", config))
        assert given_r == given_s


def test_variants_invert_only_backbone_centers() -> None:
    mol = mol_from("*[CH2:1][CH:2](C(=O)O[C@@H](C)CC)*")
    paths = strand_paths(mol, LINEAR_FACES)
    centers = backbone_stereocenters(mol, paths)
    pendant = next(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )

    plus, minus = (stereo_variant(mol, centers, config, paths) for config in (1, -1))
    assert plus.GetAtomWithIdx(centers[0]).GetChiralTag() != minus.GetAtomWithIdx(centers[0]).GetChiralTag()
    assert plus.GetAtomWithIdx(pendant).GetChiralTag() == minus.GetAtomWithIdx(pendant).GetChiralTag()


def test_variants_are_copies() -> None:
    mol = mol_from("*[CH2:1][CH:2](C)*")
    paths = strand_paths(mol, LINEAR_FACES)
    (center,) = backbone_stereocenters(mol, paths)
    stereo_variant(mol, [center], -1, paths)

    assert mol.GetAtomWithIdx(center).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_bad_config_raises() -> None:
    mol = mol_from("*[CH2:1][CH:2](C)*")
    paths = strand_paths(mol, LINEAR_FACES)
    with pytest.raises(ValueError, match="Config"):
        stereo_variant(mol, backbone_stereocenters(mol, paths), 0, paths)
