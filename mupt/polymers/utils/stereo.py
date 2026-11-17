"""
RDKit-level construction of the two stereo variants of a tactic repeat unit

A unit's config (+1 or -1) is defined relative to its BACKBONE, not to how its SMILES
happens to be written. At each backbone stereocenter, the neighbors are put in a
canonical order:
    1. the head-side backbone neighbor, then the tail-side one (strand by strand)
    2. the side group X (by default the side neighbor of highest CIP rank)
    3. the remaining neighbor (usually H)
Config +1 is counterclockwise in that order, and -1 clockwise. Equivalently, with the
backbone drawn vertically (head up) in a Fischer projection, X sits on one fixed side
for +1 and on the other for -1, so equal configs always mean "same side" -- for any
pair of units, and however the SMILES were written. That is what makes "isotactic"
meaningful between different units (e.g. styrene and methyl methacrylate).

Strand paths run from each head port's "*" linker to the matching tail port's "*"
linker, so every backbone atom (anchors included) has a head-side and a tail-side
neighbor.
"""

from typing import Collection, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import Mol

from .._common import Config, FaceSpec
from ..exceptions import FaceError
from ..rules.faces import port_atom_indices

# a config's chiral tag in canonical neighbor order
_CANONICAL_TAG = {
    1: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    -1: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}
_OPPOSITE_TAG = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}


# The backbone
def strand_paths(mol: Mol, faces: FaceSpec) -> list[list[int]]:
    """
    Atom indices along each backbone strand: head linker, head anchor ... tail anchor, tail linker

    Strand k joins the k-th port of the 'head' face to the k-th port of the 'tail' face
    """
    if not {"head", "tail"} <= set(faces):
        raise FaceError(f"A backbone needs 'head' and 'tail' faces, got faces {list(faces)}")
    if len(faces["head"]) != len(faces["tail"]):
        raise FaceError(f"Need one tail port per head port, got faces {dict(faces)}")

    ports = port_atom_indices(mol)  # label -> (anchor, linker)
    if missing := [port for face in ("head", "tail") for port in faces[face] if port not in ports]:
        raise FaceError(f"Ports {missing} aren't in the molecule")

    paths = []
    for head_port, tail_port in zip(faces["head"], faces["tail"]):
        (head_anchor, head_linker), (tail_anchor, tail_linker) = ports[head_port], ports[tail_port]
        if head_anchor == tail_anchor:
            between = [head_anchor]  # single-atom backbone, e.g. [*:1][CH](C)[*:2]
        else:
            between = list(Chem.GetShortestPath(mol, head_anchor, tail_anchor))
        paths.append([head_linker, *between, tail_linker])

    return paths


def backbone_atom_indices(paths: Sequence[Sequence[int]]) -> set[int]:
    """Real atoms on any strand path (linkers excluded)"""
    return {idx for path in paths for idx in path[1:-1]}


def backbone_stereocenters(mol: Mol, paths: Sequence[Sequence[int]]) -> list[int]:
    """Indices of potential tetrahedral stereocenters lying on the backbone"""
    backbone = backbone_atom_indices(paths)
    return sorted(
        info.centeredOn
        for info in Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True, flagPossible=True)
        if (info.type == Chem.StereoType.Atom_Tetrahedral) and (info.centeredOn in backbone)
    )


# Canonical neighbor order
def _cip_ranks(mol: Mol) -> list[int]:
    """Legacy CIP rank of every atom (higher = higher priority)"""
    ranked = Chem.Mol(mol)
    for atom in ranked.GetAtoms():
        atom.SetAtomMapNum(0)  # map numbers (port labels, overrides) would otherwise sway the ranks

    Chem.AssignStereochemistry(ranked, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    if not all(atom.HasProp("_CIPRank") for atom in ranked.GetAtoms()):
        raise RuntimeError(
            "RDKit didn't assign CIP ranks (legacy stereo perception may be off); "
            "pass side_groups to choose each unit's side group explicitly"
        )
    return [int(atom.GetProp("_CIPRank")) for atom in ranked.GetAtoms()]


def canonical_neighbors(
    mol: Mol,
    center: int,
    paths: Sequence[Sequence[int]],
    side_groups: Collection[int] = (),
    cip_ranks: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """
    A backbone stereocenter's neighbors in canonical order (see module docstring)

    side_groups optionally names atoms to use as the side group X; one that neighbors
    this center overrides the default choice of highest CIP rank
    """
    atom = mol.GetAtomWithIdx(center)
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    if len(neighbors) != 4:
        raise ValueError(f"Stereocenter {center} needs 4 explicit neighbors (add hydrogens), has {len(neighbors)}")

    # 1) backbone neighbors, head side before tail side, strand by strand
    ordered: list[int] = []
    for path in paths:
        if center in path[1:-1]:
            i = path.index(center)
            ordered += [idx for idx in (path[i - 1], path[i + 1]) if idx not in ordered]

    # 2, 3) side neighbors: X first, then the rest. They're ranked by CIP rank; ties between
    # constitutionally equivalent groups (e.g. TTSBI's two CH2 arms) are broken by closeness
    # to the head-side backbone neighbor, avoiding the center (i.e. sharing its ring)
    side = [idx for idx in neighbors if idx not in ordered]
    if side:
        ranks = cip_ranks if (cip_ranks is not None) else _cip_ranks(mol)
        distance = _distances_avoiding(mol, start=ordered[0], avoid=center) if ordered else {}

        def priority(idx: int) -> tuple[int, float]:
            return (ranks[idx], -distance.get(idx, float("inf")))

        side.sort(key=priority, reverse=True)

        overrides = [idx for idx in side if idx in side_groups]
        if overrides:
            side.remove(overrides[0])
            side.insert(0, overrides[0])
        elif (len(side) > 1) and (priority(side[0]) == priority(side[1])):
            raise ValueError(
                f"Can't choose a side group at atom {center}: its side neighbors {side} rank equally; "
                "pass side_groups to choose one"
            )
        ordered += side

    return tuple(ordered)


def _distances_avoiding(mol: Mol, start: int, avoid: int) -> dict[int, int]:
    """Bond distances from one atom to every atom reachable without passing through another"""
    distances = {start: 0}
    frontier = [start]
    while frontier:
        next_frontier = []
        for idx in frontier:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                j = nbr.GetIdx()
                if (j != avoid) and (j not in distances):
                    distances[j] = distances[idx] + 1
                    next_frontier.append(j)
        frontier = next_frontier

    return distances


def _permutation_parity(first: Sequence[int], second: Sequence[int]) -> int:
    """0 if second is an even permutation of first, 1 if odd"""
    position = {idx: i for i, idx in enumerate(first)}
    permutation = [position[idx] for idx in second]

    parity = 0
    for i in range(len(permutation)):
        while permutation[i] != i:  # sort by swaps, counting them
            j = permutation[i]
            permutation[i], permutation[j] = permutation[j], permutation[i]
            parity ^= 1
    return parity


# Building variants
def stereo_variant(
    mol: Mol,
    centers: Sequence[int],
    config: Config,
    paths: Sequence[Sequence[int]],
    side_groups: Collection[int] = (),
) -> Mol:
    """
    A copy of a Mol with every given backbone stereocenter set to config

    Any stereo the input specified at those centers is replaced; side-chain
    stereocenters are left untouched
    """
    if config not in _CANONICAL_TAG:
        raise ValueError(f"Config must be +1 or -1, got {config!r}")

    variant = Chem.Mol(mol)
    ranks = _cip_ranks(mol) if centers else None
    for idx in centers:
        canonical = canonical_neighbors(mol, idx, paths, side_groups, cip_ranks=ranks)

        # RDKit tags are relative to the order of the atom's bonds, so convert
        atom = variant.GetAtomWithIdx(idx)
        bond_order = [bond.GetOtherAtomIdx(idx) for bond in atom.GetBonds()]
        tag = _CANONICAL_TAG[config]
        if _permutation_parity(canonical, bond_order):
            tag = _OPPOSITE_TAG[tag]
        atom.SetChiralTag(tag)

    Chem.AssignStereochemistry(variant, cleanIt=False, force=True)
    return variant
