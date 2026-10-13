"""Unit tests for the tetrahedral stereocenter guard"""
# ruff: noqa: D103 ("undocumented public function")

import time
from typing import Optional

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdmolfiles import MolFromSmiles
from scipy.spatial.transform import RigidTransform, Rotation

from mupt.chemistry.sanitization import sanitized_mol
from mupt.chemistry.smiles import DEFAULT_SMILES_READ_PARAMS
from mupt.chemistry.stereo import (
    StereoInversionError,
    assert_stereo_preserved,
    atom_positions,
    atomic_bond_graph,
    capture_stereo,
    capture_stereo_from_arrays,
    check_stereo,
    normalized_volume,
    perceive_stereocenters,
)
from mupt.interfaces.rdkit import primitive_from_rdkit, suppress_rdkit_logs
from mupt.mupr.primitives import Primitive
from mupt.mupr.topology import TopologicalStructure
from mupt.geometry.shapes import PointCloud

MIRROR = np.array([-1.0, 1.0, 1.0])

# expected stereocenters (atom indices, explicit hydrogens appended after heavy atoms)
PANEL: dict[str, tuple[str, list[int]]] = {
    "CHFClBr": ("F[C@H](Cl)Br", [1]),
    "L-alanine": ("C[C@H](N)C(=O)O", [1]),
    "(R)-2-butanol": ("C[C@@H](O)CC", [1]),
    "ethanol": ("CCO", []),
    "dichloromethane": ("ClCCl", []),
    "isobutane": ("CC(C)C", []),
    "neopentane": ("CC(C)(C)C", []),
    "glycine": ("NCC(=O)O", []),
    "methylcyclohexane": ("CC1CCCCC1", []),
    "meso-tartaric acid": ("O[C@H](C(=O)O)[C@@H](O)C(=O)O", [1, 5]),
    "cis-1,4-dimethylcyclohexane": ("C[C@H]1CC[C@@H](C)CC1", [1, 4]),
    "trans-1,4-dimethylcyclohexane": ("C[C@H]1CC[C@H](C)CC1", [1, 4]),
    "cis-1,2-dimethylcyclopropane": ("C[C@H]1C[C@@H]1C", [1, 3]),
    "cis-decalin": ("[C@H]12CCCC[C@@H]1CCCC2", [0, 5]),
    "o-tolyl p-tolyl carbinol": ("OC(c1ccccc1C)c1ccc(C)cc1", [1]),
    "bis(p-tolyl) carbinol": ("OC(c1ccc(C)cc1)c1ccc(C)cc1", []),
    "menthol": ("CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", [3, 6, 9]),
    "isotactic PP pentamer": (
        "CC(C)C[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)CC(C)C",
        [4, 7, 10, 13, 16],
    ),
    "symmetric isotactic PP": ("CC[C@H](C)C[C@H](C)C[C@H](C)CC", [2, 5, 8]),
    "methyl ethyl sulfoxide": ("C[S@](=O)CC", [1]),
    "quaternary ammonium": ("C[N+](CC)(CCC)CCCC", [1]),
}


def seeded_primitive(
    smiles: str, seed: int = 20261012, label: Optional[str] = None
) -> Primitive:
    """Like primitive_from_smiles(embed_positions=True), but with a seeded embedding"""
    with suppress_rdkit_logs():
        rdmol = sanitized_mol(
            MolFromSmiles(smiles, params=DEFAULT_SMILES_READ_PARAMS), add_Hs=True
        )
        conformer_idx = EmbedMolecule(rdmol, randomSeed=seed)
        return primitive_from_rdkit(rdmol, conformer_idx=conformer_idx, label=label)


def rdkit_graph(smiles: str) -> tuple[Chem.Mol, list[str], list[tuple[int, int]]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return mol, elements, bonds


def positions_of(prim: Primitive) -> np.ndarray:
    atoms, _ = atomic_bond_graph(prim)
    return atom_positions(atoms)


# geometry
def test_normalized_volume_flips_under_reflection_only() -> None:
    center = np.zeros(3)
    neighbors = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ])
    volume = normalized_volume(center, neighbors)
    assert abs(volume) > 0.5

    assert normalized_volume(center, neighbors * MIRROR) == pytest.approx(-volume)
    rotation = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
    moved = neighbors @ rotation.T + np.array([4.0, -2.0, 7.0])
    assert normalized_volume(
        center + np.array([4.0, -2.0, 7.0]), moved
    ) == pytest.approx(volume)
    assert normalized_volume(center, 10.0 * neighbors) == pytest.approx(volume)


def test_normalized_volume_degenerate_inputs() -> None:
    assert np.isnan(normalized_volume(np.zeros(3), np.full((4, 3), np.nan)))
    assert np.isnan(normalized_volume(np.zeros(3), np.ones((3, 3))))
    assert normalized_volume(np.zeros(3), np.ones((4, 3))) == 0.0


# perception
@pytest.mark.parametrize("name", list(PANEL))
def test_perceived_stereocenters(name: str) -> None:
    smiles, expected = PANEL[name]
    mol, elements, bonds = rdkit_graph(smiles)
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    assert perceive_stereocenters(elements, bonds, charges) == expected


@pytest.mark.parametrize("name", list(PANEL))
def test_perception_agrees_with_rdkit(name: str) -> None:
    """Cross-check against the RDKit perception used by the FlowerMD stereo guard"""
    smiles, _ = PANEL[name]
    mol, elements, bonds = rdkit_graph(smiles)
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    rdkit_centers = sorted(
        int(info.centeredOn)
        for info in Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=True)
        if info.type == Chem.StereoType.Atom_Tetrahedral
    )
    assert perceive_stereocenters(elements, bonds, charges) == rdkit_centers


def test_pseudoasymmetric_center_is_kept() -> None:
    # C3 of ribaric acid ((2R,4S)-2,3,4-trihydroxyglutaric acid) is pseudoasymmetric:
    # inverting it gives xylaric acid. RDKit's FindPotentialStereo omits it here
    _, elements, bonds = rdkit_graph("OC(=O)[C@H](O)C(O)[C@@H](O)C(=O)O")
    assert perceive_stereocenters(elements, bonds) == [3, 5, 7]


def polystyrene_graph(n_repeats: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Elements and bonds of CH3-[CH2-CH(C6H5)]n-CH3, with explicit hydrogens"""
    elements: list[str] = []
    bonds: list[tuple[int, int]] = []

    def add(element: str, *bonded: int) -> int:
        elements.append(element)
        bonds.extend((atom, len(elements) - 1) for atom in bonded)
        return len(elements) - 1

    def add_hydrogens(atom: int, count: int) -> None:
        for _ in range(count):
            add("H", atom)

    previous = add("C")
    add_hydrogens(previous, 3)
    for _ in range(n_repeats):
        methylene = add("C", previous)
        add_hydrogens(methylene, 2)
        previous = add("C", methylene)
        add_hydrogens(previous, 1)
        ring = [add("C", previous)]
        for _ in range(5):
            ring.append(add("C", ring[-1]))
            add_hydrogens(ring[-1], 1)
        bonds.append((ring[0], ring[-1]))
    add_hydrogens(add("C", previous), 3)
    return elements, bonds


def test_perception_is_linear_in_chain_length() -> None:
    # quadratic perception takes minutes here, and recursive hashing overflows
    elements, bonds = polystyrene_graph(2000)
    started = time.perf_counter()
    assert len(perceive_stereocenters(elements, bonds)) == 2000
    assert time.perf_counter() - started < 10.0


def test_adapter_graph_matches_rdkit() -> None:
    smiles = PANEL["menthol"][0]
    _, elements, bonds = rdkit_graph(smiles)
    atoms, prim_bonds = atomic_bond_graph(seeded_primitive(smiles))
    assert [atom.element.symbol for atom in atoms] == elements
    assert prim_bonds == sorted(tuple(sorted(bond)) for bond in bonds)


# capture
def test_enantiomers_have_opposite_signs() -> None:
    # same atom order, opposite configuration
    sinister = capture_stereo(seeded_primitive("C[C@H](O)CC"))
    rectus = capture_stereo(seeded_primitive("C[C@@H](O)CC"))
    assert [c.atom for c in sinister.centers] == [c.atom for c in rectus.centers] == [1]
    assert sinister.centers[0].sign == -rectus.centers[0].sign


def test_sign_is_a_property_of_configuration_not_conformer() -> None:
    signs = {
        center.sign
        for seed in range(5)
        for center in capture_stereo(
            seeded_primitive(PANEL["menthol"][0], seed=seed)
        ).centers
        if center.atom == 3
    }
    assert len(signs) == 1


def test_explicit_centers_are_honored() -> None:
    prim = seeded_primitive(PANEL["menthol"][0])
    reference = capture_stereo(prim, centers=[6])
    assert [center.atom for center in reference.centers] == [6]


def test_capture_rejects_invalid_centers() -> None:
    prim = seeded_primitive("C[C@H](O)CC")
    with pytest.raises(ValueError, match="neighbors"):
        capture_stereo(prim, centers=[2])  # hydroxyl oxygen


def test_capture_rejects_planar_reference() -> None:
    elements = ["C", "F", "Cl", "Br", "H"]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    flat = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float
    )
    with pytest.raises(ValueError, match="near-planar"):
        capture_stereo_from_arrays(elements, bonds, flat)


# checking
@pytest.mark.parametrize(
    "name",
    [
        "CHFClBr",
        "meso-tartaric acid",
        "cis-1,4-dimethylcyclohexane",
        "isotactic PP pentamer",
        "methyl ethyl sulfoxide",
    ],
)
def test_reflection_inverts_every_center(name: str) -> None:
    prim = seeded_primitive(PANEL[name][0])
    reference = capture_stereo(prim)
    positions = positions_of(prim)
    assert reference.centers
    assert check_stereo(positions, reference).passed

    report = check_stereo(positions * MIRROR, reference)
    assert report.inverted == [center.atom for center in reference.centers]
    with pytest.raises(StereoInversionError) as excinfo:
        assert_stereo_preserved(positions * MIRROR, reference)
    assert excinfo.value.report.inverted == report.inverted


def test_proper_motion_preserves_every_center() -> None:
    prim = seeded_primitive(PANEL["menthol"][0])
    reference = capture_stereo(prim)
    rotation = Rotation.from_euler("zyx", [1.0, 2.0, -0.5]).as_matrix()
    moved = positions_of(prim) @ rotation.T + np.array([10.0, -3.0, 5.0])
    assert assert_stereo_preserved(moved, reference).passed


def test_swapping_substituents_inverts_only_that_center() -> None:
    prim = seeded_primitive(PANEL["meso-tartaric acid"][0])
    reference = capture_stereo(prim)
    positions = positions_of(prim)
    center = reference.centers[0]
    first, second = center.neighbors[0], center.neighbors[1]
    # reflect the center's bond vectors to two neighbors through the plane
    # bisecting them, which exchanges their directions
    swapped = positions.copy()
    swapped[first], swapped[second] = positions[second], positions[first]
    report = check_stereo(swapped, reference)
    assert report.inverted == [center.atom]
    assert report.statuses[reference.centers[1].atom] == "preserved"


def test_flattened_and_nonfinite_centers_are_reported() -> None:
    prim = seeded_primitive("C[C@H](O)CC")
    reference = capture_stereo(prim)
    atom = reference.centers[0].atom
    positions = positions_of(prim)

    flattened = positions.copy()
    flattened[list(reference.centers[0].neighbors), 2] = flattened[atom, 2]
    flattened[atom] = flattened[list(reference.centers[0].neighbors)].mean(axis=0)
    report = check_stereo(flattened, reference)
    assert report.near_planar == [atom] and not report.passed

    broken = positions.copy()
    broken[atom] = np.nan
    assert check_stereo(broken, reference).nonfinite == [atom]


def test_wrapped_coordinates_need_box() -> None:
    prim = seeded_primitive("C[C@H](O)CC")
    reference = capture_stereo(prim)
    box = np.array([12.0, 12.0, 12.0])
    wrapped = positions_of(prim)
    wrapped[reference.centers[0].neighbors[0]] += box * np.array([1.0, 0.0, -1.0])
    assert check_stereo(wrapped, reference, box_lengths=box).passed


def test_check_against_primitive() -> None:
    prim = seeded_primitive(PANEL["menthol"][0])
    reference = capture_stereo(prim)
    assert check_stereo(prim, reference).passed

    atoms, _ = atomic_bond_graph(prim)
    for atom in atoms:
        atom.shape = PointCloud(positions=np.asarray(atom.shape.centroid) * MIRROR)
    assert len(check_stereo(prim, reference).inverted) == 3

    with pytest.raises(ValueError, match="do not match"):
        check_stereo(seeded_primitive("C[C@H](O)CC"), reference)


# centers spanning residues
RESIDUE_SMILES = {
    "head": "CC[C@H:2](C)-*",
    "mid": "*-[CH2:1][C@H:2](C)-*",
    "mid_inverted": "*-[CH2:1][C@@H:2](C)-*",
    "tail": "*-[CH2:1]C(C)C",
}


def polypropylene_chain(sequence: list[str]) -> Primitive:
    """A chain of polypropylene residue templates, bonded but not placed"""
    templates = {
        name: seeded_primitive(RESIDUE_SMILES[name], label=name)
        for name in set(sequence)
    }
    chain = Primitive(label="chain")
    handles = [chain.attach_child(templates[name].copy()) for name in sequence]
    chain.set_topology(
        nx.path_graph(handles, create_using=TopologicalStructure),
        max_registration_iter=100,
    )
    return chain


def junction_centers(chain: Primitive) -> list[int]:
    """Recorded centers which have a neighbor in another residue"""
    atoms, _ = atomic_bond_graph(chain)
    reference = capture_stereo(chain)
    return [
        center.atom
        for center in reference.centers
        if any(
            atoms[n].parent is not atoms[center.atom].parent for n in center.neighbors
        )
    ]


def test_junction_centers_are_captured_from_residue_templates() -> None:
    chain = polypropylene_chain(["head", "mid", "mid", "mid", "tail"])
    reference = capture_stereo(chain)
    assert len(reference.centers) == 4
    # every backbone CH of a mid residue has its fourth neighbor in the next residue
    assert len(junction_centers(chain)) == 4
    # isotactic: every residue template defines the same handedness
    assert len({center.sign for center in reference.centers[1:]}) == 1


def test_junction_handedness_follows_residue_template() -> None:
    iso = capture_stereo(polypropylene_chain(["head", "mid", "mid", "tail"]))
    syndio = capture_stereo(
        polypropylene_chain(["head", "mid", "mid_inverted", "tail"])
    )
    assert [c.atom for c in iso.centers] == [c.atom for c in syndio.centers]
    assert iso.centers[1].sign == syndio.centers[1].sign
    assert iso.centers[2].sign == -syndio.centers[2].sign


def test_moving_residues_rigidly_keeps_junction_handedness() -> None:
    chain = polypropylene_chain(["head", "mid", "mid", "mid", "tail"])
    before = capture_stereo(chain)
    rng = np.random.default_rng(7)
    for residue in chain.children:
        residue.rigidly_transform(
            RigidTransform.from_components(
                rotation=Rotation.random(random_state=rng),
                translation=rng.uniform(-20, 20, size=3),
            )
        )
    after = capture_stereo(chain)
    assert [c.sign for c in after.centers] == [c.sign for c in before.centers]
