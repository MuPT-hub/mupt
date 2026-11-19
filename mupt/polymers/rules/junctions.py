"""
Junction rules: deciding which ports bond across each edge of a descriptive graph

A junction joins one face of an upstream unit u to one face of a downstream unit v,
written (u, face_u, v, face_v). The generator decides WHICH faces meet; a junction rule
decides HOW their strands pair up, then writes the resulting bonds onto the edge.

A pairing p means strand k of face_u bonds to strand p[k] of face_v, e.g. for width 2:
    (0, 1) : straight junction (port order kept)
    (1, 0) : flipped junction (port order reversed)

Writing a new rule usually means overriding pairings() only; override assign() when a
rule must decide more than strand order (e.g. which faces or ports to use).
"""

from typing import Hashable, Mapping, Sequence
from abc import ABC, abstractmethod
import warnings

import numpy as np
from networkx import Graph
from rdkit import Chem
from rdkit.Chem import Mol

from .._common import EDGE_BONDS, NODE_UNIT, Bond, FaceSpec, Junction, Node, Pairing, UnitLabel
from ..exceptions import FaceError, TopologyValidationError


# Helpers shared by all rules
def is_permutation(pairing: Sequence[int], width: int) -> bool:
    """Whether a pairing is a permutation of range(width)"""
    return sorted(pairing) == list(range(width))


def faces_of_node(G: Graph, node: Node, faces: Mapping[Hashable, FaceSpec]) -> FaceSpec:
    """
    Look up the face spec of the unit a node represents

    Faces are keyed by unit label; a generated unit key (label, config) is looked up
    by its label, while any other key (e.g. a plain string) is looked up as is
    """
    key = G.nodes[node][NODE_UNIT]
    if key in faces:
        return faces[key]
    if isinstance(key, tuple) and key and (key[0] in faces):
        return faces[key[0]]

    raise FaceError(f"No faces given for unit {key!r} of node {node!r}")


def junction_widths(
    G: Graph,
    junctions: Sequence[Junction],
    faces: Mapping[Hashable, FaceSpec],
) -> list[int]:
    """Number of bonds at each junction; both faces of a junction must have the same width"""
    widths = []
    for u, face_u, v, face_v in junctions:
        ports_u = faces_of_node(G, u, faces)[face_u]
        ports_v = faces_of_node(G, v, faces)[face_v]
        if len(ports_u) != len(ports_v):
            raise FaceError(
                f"Can't join face {face_u!r} of node {u!r} ({len(ports_u)} ports) "
                f"to face {face_v!r} of node {v!r} ({len(ports_v)} ports)"
            )
        widths.append(len(ports_u))

    return widths


# The framework
class JunctionRule(ABC):
    """
    Abstract base class for deciding how the faces at each junction bond together

    Generators call assign() once per chain, passing junctions in growth order
    (head -> tail along chains), so "upstream" never depends on node numbering
    """

    def validate(self, faces: Mapping[UnitLabel, FaceSpec]) -> None:
        """Check the rule against every unit's faces when a generator is built (default: no-op)"""

    @abstractmethod
    def pairings(
        self,
        G: Graph,
        junctions: Sequence[Junction],
        faces: Mapping[UnitLabel, FaceSpec],
        rng: np.random.Generator,
    ) -> list[Pairing]:
        """
        Choose one pairing per junction, in order

        All of a chain's junctions are passed at once, so rules can draw random
        numbers in one vectorized call; junction_widths() gives each junction's width
        """
        ...

    def assign(
        self,
        G: Graph,
        junctions: Sequence[Junction],
        faces: Mapping[UnitLabel, FaceSpec],
        rng: np.random.Generator,
    ) -> None:
        """Write the bonds of every junction onto its edge, as G.edges[u, v]['bonds']"""
        pairings = self.pairings(G, junctions, faces, rng)
        if len(pairings) != len(junctions):
            raise ValueError(f"{self!r} gave {len(pairings)} pairings for {len(junctions)} junctions")

        for (u, face_u, v, face_v), pairing in zip(junctions, pairings):
            if not G.has_edge(u, v):
                raise TopologyValidationError(f"Junction ({u!r}, {v!r}) isn't an edge of the graph")

            ports_u = faces_of_node(G, u, faces)[face_u]
            ports_v = faces_of_node(G, v, faces)[face_v]
            if len(ports_u) != len(ports_v):
                raise FaceError(f"Faces {face_u!r} and {face_v!r} at junction ({u!r}, {v!r}) differ in width")
            if not is_permutation(pairing, len(ports_u)):
                raise ValueError(f"{self!r} gave pairing {pairing} for a width-{len(ports_u)} junction")

            # strand k of u's face bonds to strand pairing[k] of v's face
            bonds: tuple[Bond, ...] = tuple(
                ((u, ports_u[k]), (v, ports_v[j]))
                for k, j in enumerate(pairing)
            )
            G.edges[u, v][EDGE_BONDS] = bonds


# Implementations
class FixedPairing(JunctionRule):
    """Every junction is straight: strand k bonds to strand k (uses no random numbers)"""

    def pairings(self, G, junctions, faces, rng) -> list[Pairing]:
        return [tuple(range(width)) for width in junction_widths(G, junctions, faces)]

    def __repr__(self) -> str:
        return "FixedPairing()"


class RandomFlip(JunctionRule):
    """Each junction is flipped (strand order reversed) independently, with probability p"""

    def __init__(self, p: float) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Flip probability must be in [0, 1], got {p}")
        self.p = p

    def validate(self, faces: Mapping[UnitLabel, FaceSpec]) -> None:
        widths = {len(face) for spec in faces.values() for face in spec.values()}
        if widths <= {1}:
            warnings.warn("RandomFlip has no effect: every face has a single port, so there's nothing to flip")

    def pairings(self, G, junctions, faces, rng) -> list[Pairing]:
        widths = junction_widths(G, junctions, faces)

        # exactly one draw per junction, whatever its width, so the random stream
        # (and every later draw from it) doesn't depend on the units' widths
        flips = rng.random(len(junctions)) < self.p

        return [
            tuple(reversed(range(width))) if flip else tuple(range(width))
            for width, flip in zip(widths, flips)
        ]

    def __repr__(self) -> str:
        return f"RandomFlip(p={self.p})"


# Symmetry heuristic (RDKit level; called by generators, so rules stay chemistry-free)
def anchors_symmetry_equivalent(mol: Mol, anchor_idxs: Sequence[int]) -> bool:
    """
    Whether the given anchor atoms all share one symmetry class of a Mol (map numbers ignored)

    A heuristic for whether flipping a junction on this side can produce a distinct molecule;
    True means a flip MAY be chemically degenerate, not that it certainly is
    """
    unmapped = Chem.Mol(mol)
    for atom in unmapped.GetAtoms():
        atom.SetAtomMapNum(0)  # map numbers would otherwise make every port look unique

    ranks = list(Chem.CanonicalRankAtoms(unmapped, breakTies=False))
    return len({ranks[idx] for idx in anchor_idxs}) == 1
