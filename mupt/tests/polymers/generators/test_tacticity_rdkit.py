"""
End-to-end check that built chains have the tacticity their configs claim

Each chain is exported to RDKit and every interior diad is judged meso or racemo by
symmetry, WITHOUT using the code that sets configurations:

    - keep the diad's two backbone stereocenters, the backbone between them, and two
      backbone atoms outward on each side
    - reduce each center's side group X to Cl and its other substituent to H
      (X chosen like the default: an aromatic C, else a C bonded to O, else the only one)
    - the reduced diad, e.g. Et-CH(Cl)-CH2-CH(Cl)-Et, is meso exactly when it's achiral,
      i.e. when it's identical to its mirror image

Reducing both centers the same way makes diads between DIFFERENT monomers (e.g.
styrene-MMA) comparable too, which is where SMILES-dependent configs used to go wrong.
"""

import pytest
from rdkit import Chem

from mupt.interfaces.rdkit import rdkit_mol_from_primitive
from mupt.polymers.generators.linear import LinearPolymerGenerator
from mupt.polymers.models import (
    SequenceAlternating,
    TacticityBernoullian,
    TacticityConstant,
    TacticityEnantiomorphicSite,
    atactic,
    isotactic,
    syndiospecific_site,
    syndiotactic,
)
from mupt.polymers.utils.realize import universe_from_descriptive_graphs

N_UNITS = 8
N_CHAINS = 3

# monomers, each written two ways (different atom orders, same structure)
PP = "*[CH2:1][CH:2](C)*"
STYRENE = {"A": "*[CH2:1][CH:2](c1ccccc1)*", "B": "*[CH2:1][CH:2](*)c1ccccc1"}
MMA = {"A": "*[CH2:1][C:2](C)(C(=O)OC)*", "B": "*[CH2:1][C:2](C(=O)OC)(C)*"}

SYSTEMS = {
    "PP": dict(units={"PP": PP}),
    "PS": dict(units={"S": STYRENE["A"]}),
    "PMMA": dict(units={"M": MMA["A"]}),
    **{
        f"S({s})-M({m})": dict(units={"S": STYRENE[s], "M": MMA[m]}, sequence=SequenceAlternating(["S", "M"]))
        for s in "AB"
        for m in "AB"
    },
}
PRESETS = {
    "isotactic": isotactic(),
    "syndiotactic": syndiotactic(),
    "atactic": atactic(),
    "constant": TacticityConstant(),
    "TacticityBernoullian(0.7)": TacticityBernoullian(0.7),
    "TacticityEnantiomorphicSite(0.8)": TacticityEnantiomorphicSite(0.8),  # racemic site
    "TacticityEnantiomorphicSite(0.8, +1)": TacticityEnantiomorphicSite(0.8, site_handedness=1),  # enantiopure site
    "syndiospecific_site()": syndiospecific_site(),
}


# Reading tacticity off an RDKit molecule
def _side_group(mol: Chem.Mol, side: list[int]) -> int:
    """The side group X among a center's side neighbors: aromatic C, else C bonded to O, else the only one"""
    def preference(idx: int) -> tuple[bool, bool]:
        atom = mol.GetAtomWithIdx(idx)
        return (atom.GetIsAromatic(), any(nbr.GetSymbol() == "O" for nbr in atom.GetNeighbors()))

    return max(side, key=preference)


def _set_element(atom: Chem.Atom, atomic_num: int) -> None:
    """Turn an atom into a plain element (keeping its bonds, and so any neighbor's stereo parity)"""
    atom.SetAtomicNum(atomic_num)
    atom.SetIsAromatic(False)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(True)


def _is_achiral(smiles: str) -> bool:
    """Whether a molecule is identical to its mirror image"""
    mol = Chem.MolFromSmiles(smiles)
    mirror = Chem.Mol(mol)
    for atom in mirror.GetAtoms():
        atom.InvertChirality()
    return Chem.MolToSmiles(mol) == Chem.MolToSmiles(mirror)


def diads_from_rdkit(chain) -> dict[int, str]:
    """
    m/r of each interior diad of a Br...I capped chain, keyed by diad index (diad k joins units k, k+1)

    Diads within two backbone atoms of a chain end are skipped: the mirror test needs
    the same truncation on both sides
    """
    mol = Chem.RemoveHs(rdkit_mol_from_primitive(chain))
    head = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Br")
    tail = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "I")

    full_path = list(Chem.GetShortestPath(mol, head, tail))  # backbone, head to tail
    on_backbone = set(full_path)  # the caps count as backbone, never as side groups
    path = full_path[1:-1]
    centers = [
        idx for idx in path
        if any(nbr.GetIdx() not in on_backbone for nbr in mol.GetAtomWithIdx(idx).GetNeighbors())
    ]

    diads: dict[int, str] = {}
    for k, (first, second) in enumerate(zip(centers, centers[1:])):
        i, j = path.index(first), path.index(second)
        if (i < 2) or (j + 2 >= len(path)):
            continue

        # reduce both centers to CH(Cl), keeping each one's configuration
        reduced = Chem.RWMol(mol)
        keep = set(path[i - 2: j + 3])
        for center in (first, second):
            side = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(center).GetNeighbors() if nbr.GetIdx() not in on_backbone]
            x = _side_group(mol, side)
            for idx in side:
                _set_element(reduced.GetAtomWithIdx(idx), 17 if (idx == x) else 1)
            keep.update(side)

        fragment = Chem.MolFragmentToSmiles(reduced, atomsToUse=sorted(keep), isomericSmiles=True)
        diads[k] = "m" if _is_achiral(fragment) else "r"

    return diads


def diads_from_configs(G, n_units: int) -> list[str]:
    """m/r of every diad, from the configs in a chain's descriptive graph"""
    configs = [G.nodes[i]["unit"][1] for i in range(n_units)]
    return ["m" if a == b else "r" for a, b in zip(configs, configs[1:])]


# The checks
@pytest.fixture(scope="module", params=list(SYSTEMS), ids=list(SYSTEMS))
def built_chains(request) -> dict[str, list]:
    """For one system, (descriptive graph, realized chain) pairs under every preset"""
    chains = {}
    for name, preset in PRESETS.items():
        generator = LinearPolymerGenerator(size=N_UNITS, end_groups={"head": "[Br]*", "tail": "[I]*"}, tacticity=preset, **SYSTEMS[request.param])
        graphs = generator.generate_topologies(N_CHAINS, seed=0)
        universe = universe_from_descriptive_graphs(graphs, generator.lexicon)
        chains[name] = list(zip(graphs, universe.children))
    return chains


@pytest.mark.parametrize("preset", list(PRESETS))
def test_built_chains_match_their_configs(built_chains, preset) -> None:
    for G, chain in built_chains[preset]:
        expected = diads_from_configs(G, N_UNITS)
        found = diads_from_rdkit(chain)

        assert found, "no interior diads to check"
        assert found == {k: expected[k] for k in found}


@pytest.mark.parametrize(
    "preset, diad",
    [("isotactic", "m"), ("constant", "m"), ("syndiotactic", "r"), ("syndiospecific_site()", "r")],
)
def test_regular_presets_give_regular_chains(built_chains, preset, diad) -> None:
    for _, chain in built_chains[preset]:
        assert set(diads_from_rdkit(chain).values()) == {diad}


def test_atactic_chains_mix_diads(built_chains) -> None:
    found = {d for _, chain in built_chains["atactic"] for d in diads_from_rdkit(chain).values()}
    assert found == {"m", "r"}
