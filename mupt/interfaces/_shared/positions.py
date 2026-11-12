"""
Where exporters place a Primitive that stands in for several atoms
(e.g. a coarse-grained bead), shared so every interface agrees.
"""

from typing import Literal, Optional, get_args

import numpy as np

from ...mupr.primitives.primitive import Primitive
from ...mupr.primitives.predicates import is_atom
from ...trees.subselect import primoprogenitors

BeadPosition = Literal["centroid", "center_of_mass"]
"""
How to place a Primitive in space:

- ``"centroid"``: the geometric center of its shape (for a PointCloud of atoms,
  the plain average of the atom positions); if it has no shape, the plain
  average of its atoms' positions.
- ``"center_of_mass"``: the mass-weighted average of its atoms' positions; if
  any of its atoms has no position, its centroid.

For a single atom both give the atom's position.
"""

BEAD_POSITIONS: tuple[str, ...] = get_args(BeadPosition)


def check_bead_position(bead_position: str) -> None:
    """Raise ValueError unless bead_position is one of BEAD_POSITIONS"""
    if bead_position not in BEAD_POSITIONS:
        raise ValueError(
            f"bead_position must be one of {BEAD_POSITIONS}, not {bead_position!r}"
        )


def bead_position_of(
    prim: Primitive,
    bead_position: BeadPosition = "centroid",
) -> Optional[np.ndarray]:
    """
    Position of a Primitive, as chosen by `bead_position` (see BeadPosition).
    Returns None if the Primitive and its atoms have no positions.
    """
    check_bead_position(bead_position)
    atoms = list(primoprogenitors(prim, is_atom))
    atoms_positioned = bool(atoms) and all(atom.shape is not None for atom in atoms)

    if bead_position == "center_of_mass" and atoms_positioned:
        positions = np.array([atom.shape.centroid for atom in atoms], dtype=float)
        masses = np.array([atom.mass for atom in atoms], dtype=float)
        return (masses[:, None] * positions).sum(axis=0) / masses.sum()

    if prim.shape is not None:
        return np.asarray(prim.shape.centroid, dtype=float)

    if atoms_positioned:
        positions = np.array([atom.shape.centroid for atom in atoms], dtype=float)
        return positions.mean(axis=0)

    return None
