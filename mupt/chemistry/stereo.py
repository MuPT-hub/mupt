"""
Utilities for handling stereochemistry, including
CIP assignment and enumeration of stereoisomers

Also provides a guard which checks that a coordinate builder does not invert
any tetrahedral stereocenter present in its starting coordinates. The guard
reads the handedness of each center from a starting geometry (e.g. one embedded
from SMILES) and compares later geometries of the same atoms against it;
Primitives never store stereocenters or labels.

Handedness of a center whose neighbors are a, b, c, d (in atom order) is the
sign of ``det(a - d, b - d, c - d)``, divided by the three edge lengths so it
is scale free. Reflections flip the sign; rotations and translations keep it.
Centers with a lone pair in place of a fourth neighbor use the center as d.
"""

from typing import Iterable, Mapping, Optional, Sequence, Union
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from anytree import PreOrderIter

from ..mupr.primitives import Primitive


# TB DEVNOTE: just doing a kitchen sink import for now so I remember
# later what RDKit has to offer here for comprehensive documentation.
# See https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops

STEREOINFO_ATTRS: tuple[str, ...] = (
    "NOATOM",
    "centeredOn",
    "controllingAtoms",
    "descriptor",
    "permutation",
    "specified",
    "type",
)

DEFAULT_PLANAR_TOLERANCE: float = 0.05
# Elements whose three-coordinate, pyramidal centers hold their configuration.
# Three-coordinate nitrogen is left out, since amines invert at room temperature
LONE_PAIR_ELEMENTS: frozenset[str] = frozenset({"P", "S", "As", "Se"})


@dataclass(frozen=True)
class Stereocenter:
    """
    A center atom, its bonded neighbors (ascending), and the sign and
    normalized volume of its neighbor tetrahedron in the reference geometry
    """

    atom: int
    neighbors: tuple[int, ...]
    sign: int
    volume: float


@dataclass(frozen=True)
class StereoReference:
    """Element symbols (in flattened atom order) and stereocenters of a geometry"""

    elements: tuple[str, ...]
    centers: tuple[Stereocenter, ...]


@dataclass(frozen=True)
class StereoReport:
    """
    Status of every recorded center in a checked geometry: "preserved",
    "inverted", "near_planar" (normalized volume below the planar
    tolerance) or "nonfinite" (NaN or infinite coordinates)
    """

    statuses: dict[int, str] = field(default_factory=dict)
    volumes: dict[int, Optional[float]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every recorded center kept its handedness"""
        return all(status == "preserved" for status in self.statuses.values())

    def with_status(self, status: str) -> list[int]:
        """Center atoms which received a given status"""
        return [atom for atom, stat in self.statuses.items() if stat == status]

    inverted = property(lambda self: self.with_status("inverted"))
    near_planar = property(lambda self: self.with_status("near_planar"))
    nonfinite = property(lambda self: self.with_status("nonfinite"))


class StereoInversionError(RuntimeError):
    """Raised when a geometry does not keep its reference stereochemistry"""

    def __init__(self, report: StereoReport) -> None:
        self.report = report
        super().__init__(
            f"Stereochemistry not preserved: inverted at atoms {report.inverted}, "
            f"near-planar at atoms {report.near_planar}, "
            f"nonfinite at atoms {report.nonfinite}"
        )


# Geometry
def normalized_volume(
    center: ArrayLike,
    neighbors: ArrayLike,
    box_lengths: Optional[Sequence[float]] = None,
) -> float:
    """
    Signed volume of the tetrahedron of four neighbor positions (shape (4, 3))
    around a center, divided by its three edge lengths; a value in [-1, 1]

    If orthorhombic ``box_lengths`` are given, neighbor vectors are taken as
    minimum images relative to the center. Returns NaN for nonfinite input
    """
    vectors = np.asarray(neighbors, dtype=float) - np.asarray(center, dtype=float)
    if vectors.shape != (4, 3) or not np.isfinite(vectors).all():
        return float("nan")
    if box_lengths is not None:
        box = np.asarray(box_lengths, dtype=float)
        vectors -= box * np.rint(vectors / box)
    edges = vectors[:3] - vectors[3]
    denominator = float(np.prod(np.linalg.norm(edges, axis=1)))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(np.linalg.det(edges) / denominator)


def _center_volume(
    atom: int,
    neighbors: Sequence[int],
    positions: np.ndarray,
    box_lengths: Optional[Sequence[float]] = None,
    neighbor_positions: Optional[Mapping[tuple[int, int], ArrayLike]] = None,
) -> float:
    """
    Normalized volume of a center, using the center itself in place of a lone pair;
    ``neighbor_positions[atom, neighbor]`` overrides a neighbor's position where present
    """
    neighbor_positions = neighbor_positions or {}
    points = [neighbor_positions.get((atom, nb), positions[nb]) for nb in neighbors]
    if len(points) == 3:
        points.append(positions[atom])
    return normalized_volume(positions[atom], np.asarray(points), box_lengths)


# Topology
def _flatten(
    prim: Primitive,
) -> tuple[list[Primitive], list[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    """
    Atoms, bonds, and, keyed by (i, j), the linker position
    of the Connector on atom i which forms the bond to atom j
    """
    # imported here to avoid a circular import between chemistry and interfaces
    from ..interfaces._shared.topology import resolve_to_atom_cached

    atoms = [leaf for leaf in prim.leaves if leaf.is_atom]
    index_of = {id(atom): idx for idx, atom in enumerate(atoms)}
    cache: dict = {}
    bonds: set[tuple[int, int]] = set()
    linker_positions: dict[tuple[int, int], np.ndarray] = {}
    for node in PreOrderIter(prim):
        for conn_ref1, conn_ref2 in node.internal_connections:
            i = index_of[id(resolve_to_atom_cached(node, conn_ref1, cache))]
            j = index_of[id(resolve_to_atom_cached(node, conn_ref2, cache))]
            bonds.add((min(i, j), max(i, j)))
            for center, neighbor, conn_ref in ((i, j, conn_ref1), (j, i, conn_ref2)):
                child = node.fetch_child(conn_ref.primitive_handle)
                linker = child.connector_trace(conn_ref.connector_handle)[-1].linker
                if linker.position is not None:
                    linker_positions[center, neighbor] = np.asarray(linker.position)
    return atoms, sorted(bonds), linker_positions


def atomic_bond_graph(prim: Primitive) -> tuple[list[Primitive], list[tuple[int, int]]]:
    """
    Flatten a Primitive hierarchy to its atomic leaves (in pre-order) and the
    sorted pairs of atom indices bonded at any level of the hierarchy, so bonds
    between residues or other groupings are included alongside bonds within them
    """
    atoms, bonds, _ = _flatten(prim)
    return atoms, bonds


def atom_positions(atoms: Iterable[Primitive]) -> np.ndarray:
    """Positions (centroids of shapes) of a sequence of atomic Primitives"""
    positions = []
    for idx, atom in enumerate(atoms):
        if atom.shape is None:
            raise ValueError(f"Atom {idx} ({atom.label}) has no shape or position")
        positions.append(atom.shape.centroid)
    return np.array(positions, dtype=float).reshape(-1, 3)


def _bond_graph(
    elements: Sequence[str],
    bonds: Iterable[tuple[int, int]],
    charges: Optional[Sequence[int]] = None,
) -> nx.Graph:
    """Bond graph on atom indices, with nodes labelled by element and formal charge"""
    graph = nx.Graph()
    graph.add_nodes_from(
        (idx, {"label": f"{element}{charge:+d}"})
        for idx, (element, charge) in enumerate(
            zip(elements, charges or [0] * len(elements))
        )
    )
    graph.add_edges_from(bonds)
    return graph


def _symmetry_classes(graph: nx.Graph) -> dict[int, str]:
    """
    Weisfeiler-Lehman class of every atom, refined until classes stop splitting;
    atoms in different classes are not related by any symmetry of the graph
    """
    iterations = 4
    while True:
        hashes = nx.weisfeiler_lehman_subgraph_hashes(
            graph, node_attr="label", iterations=iterations
        )
        if len({h[-1] for h in hashes.values()}) == len({
            h[-2] for h in hashes.values()
        }):
            return {atom: h[-1] for atom, h in hashes.items()}
        iterations *= 2


def _is_tetrahedral(element: str, n_neighbors: int) -> bool:
    """Whether an atom can be a tetrahedral center, given its element and degree"""
    return n_neighbors == 4 or (n_neighbors == 3 and element in LONE_PAIR_ELEMENTS)


def perceive_stereocenters(
    elements: Sequence[str],
    bonds: Iterable[tuple[int, int]],
    charges: Optional[Sequence[int]] = None,
) -> list[int]:
    """
    Atoms whose inversion would change the molecule, found from the bond graph alone

    Candidates are atoms with four explicit neighbors, or three for elements in
    `LONE_PAIR_ELEMENTS`. A candidate is a stereocenter if no two of its
    neighbors are equivalent. A four-coordinate atom with one equivalent pair
    is also kept when swapping the pair is not a symmetry of the molecule:

    - the pair closes a ring through the atom, and that ring holds another such
      atom (cis/trans ring centers, e.g. C1 and C4 of 1,4-dimethylcyclohexane)
    - the pair heads two separate branches containing stereocenters
      (pseudoasymmetric centers, e.g. C3 of ribaric acid, or the middle
      backbone carbon of a symmetric tactic chain)

    When in doubt an atom is kept, since a missed inversion is worse
    than a false alarm. Hydrogens must be explicit
    """
    graph = _bond_graph(elements, bonds, charges)
    classes = _symmetry_classes(graph)

    centers: set[int] = set()
    ring_candidates: dict[int, set[int]] = {}  # atom -> atoms beyond its tied pair
    pseudo_candidates: dict[int, set[int]] = {}  # atom -> atoms of its tied branches
    for atom in graph:
        neighbors = list(graph[atom])
        if not _is_tetrahedral(elements[atom], len(neighbors)):
            continue
        counts = Counter(classes[neighbor] for neighbor in neighbors)
        ties = [nb for nb in neighbors if counts[classes[nb]] > 1]
        if not ties:
            centers.add(atom)
        elif len(neighbors) == 4 and len(ties) == 2 and graph.degree[ties[0]] > 1:
            # (equivalent terminal atoms, as in CH2 or CF2, are always harmless to swap)
            first, second = ties
            beyond = nx.restricted_view(graph, [atom], [])
            first_branch = nx.node_connected_component(beyond, first)
            if second in first_branch:
                ring_candidates[atom] = first_branch
            else:
                pseudo_candidates[atom] = first_branch | nx.node_connected_component(
                    beyond, second
                )

    for atom, ring_atoms in ring_candidates.items():
        if any(other != atom and other in ring_atoms for other in ring_candidates):
            centers.add(atom)
    protected = centers | set(ring_candidates) | set(pseudo_candidates)
    for atom, branch_atoms in pseudo_candidates.items():
        if branch_atoms & protected:
            centers.add(atom)

    return sorted(centers)


# Capture and checking
def capture_stereo_from_arrays(
    elements: Sequence[str],
    bonds: Iterable[tuple[int, int]],
    positions: ArrayLike,
    centers: Optional[Iterable[int]] = None,
    charges: Optional[Sequence[int]] = None,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
    neighbor_positions: Optional[Mapping[tuple[int, int], ArrayLike]] = None,
) -> StereoReference:
    """
    Record the handedness of the stereocenters in a starting geometry

    Parameters
    ----------
    elements : sequence of str
        Element symbol of every atom
    bonds : iterable of (int, int)
        Bonded atom index pairs
    positions : array of shape (N, 3)
        Starting coordinates
    centers : iterable of int, optional
        Atoms to record. If omitted, centers are found with `perceive_stereocenters`
    charges : sequence of int, optional
        Formal charges, used only when perceiving centers
    planar_tolerance : float, default 0.05
        Centers whose normalized volume is below this cannot be recorded
    neighbor_positions : mapping of (center, neighbor) to position, optional
        Where a neighbor sits as seen from a given center, in place of
        its entry in ``positions`` (see `capture_stereo`)
    """
    if not 0 < planar_tolerance < 1:
        raise ValueError("planar_tolerance must be between 0 and 1")
    bonds = list(bonds)
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(elements), 3):
        raise ValueError(f"Expected positions of shape ({len(elements)}, 3)")
    graph = _bond_graph(elements, bonds)
    if centers is None:
        centers = perceive_stereocenters(elements, bonds, charges)

    records = []
    for atom in sorted(set(centers)):
        neighbors = tuple(sorted(graph[atom]))
        if not _is_tetrahedral(elements[atom], len(neighbors)):
            raise ValueError(
                f"Atom {atom} ({elements[atom]}) has {len(neighbors)} neighbors; "
                "a tetrahedral center needs four, or three for "
                f"{sorted(LONE_PAIR_ELEMENTS)} (add explicit hydrogens)"
            )
        volume = _center_volume(
            atom, neighbors, positions, neighbor_positions=neighbor_positions
        )
        if not abs(volume) >= planar_tolerance:  # also catches NaN
            raise ValueError(
                f"Atom {atom} has a nonfinite or near-planar reference geometry "
                f"(normalized volume {volume:.3g}), so its handedness is undefined. "
                "Fix the starting coordinates, or pass `centers` to leave it out."
            )
        records.append(Stereocenter(atom, neighbors, 1 if volume > 0 else -1, volume))
    return StereoReference(elements=tuple(elements), centers=tuple(records))


def capture_stereo(
    prim: Primitive,
    centers: Optional[Iterable[int]] = None,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
    use_connector_positions: bool = True,
) -> StereoReference:
    """
    Record the handedness of the stereocenters in a Primitive's current coordinates

    The hierarchy is flattened to atoms and bonds first (see `atomic_bond_graph`),
    so centers whose neighbors lie in different residues are found like any other.

    By default each neighbor of a center is taken to sit at the linker of the
    center atom's Connector for that bond. Within a residue the linker coincides
    with the neighbor atom; across residues it is where the residue's own
    template put its bond partner (the ``*`` of a residue SMILES). A center at a
    junction thus gets the handedness its residue template defines, even before
    residues are placed relative to one another. Pass
    ``use_connector_positions=False`` to use atom positions only, e.g. for
    coordinates written by a builder which moves atoms but not Connectors.
    ``centers`` and ``planar_tolerance`` are as in `capture_stereo_from_arrays`

    Examples
    --------
    >>> from mupt.interfaces.smiles import primitive_from_smiles
    >>> from mupt.chemistry.stereo import capture_stereo, check_stereo
    >>> prim = primitive_from_smiles("C[C@H](O)CC", embed_positions=True)
    >>> reference = capture_stereo(prim)
    >>> [center.atom for center in reference.centers]
    [1]
    >>> check_stereo(prim, reference).passed
    True
    """
    atoms, bonds, linker_positions = _flatten(prim)
    return capture_stereo_from_arrays(
        elements=[atom.element.symbol for atom in atoms],
        bonds=bonds,
        positions=atom_positions(atoms),
        centers=centers,
        charges=[int(getattr(atom.element, "charge", 0)) for atom in atoms],
        planar_tolerance=planar_tolerance,
        neighbor_positions=linker_positions if use_connector_positions else None,
    )


def check_stereo(
    target: Union[Primitive, ArrayLike],
    reference: StereoReference,
    box_lengths: Optional[Sequence[float]] = None,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
) -> StereoReport:
    """
    Compare a geometry against the handedness recorded in a StereoReference

    ``target`` is either a Primitive with the same atoms as the reference
    or an (N, 3) array of positions in flattened atom order. Pass orthorhombic
    ``box_lengths`` for coordinates wrapped into a periodic box
    """
    if isinstance(target, Primitive):
        atoms = [leaf for leaf in target.leaves if leaf.is_atom]
        if tuple(atom.element.symbol for atom in atoms) != reference.elements:
            raise ValueError("Atoms of the target do not match the reference")
        target = atom_positions(atoms)
    positions = np.asarray(target, dtype=float)
    if positions.shape != (len(reference.elements), 3):
        raise ValueError(f"Expected positions of shape ({len(reference.elements)}, 3)")

    report = StereoReport()
    for center in reference.centers:
        volume = _center_volume(center.atom, center.neighbors, positions, box_lengths)
        if not np.isfinite(volume):
            status, volume = "nonfinite", None
        elif abs(volume) < planar_tolerance:
            status = "near_planar"
        else:
            status = "preserved" if volume * center.sign > 0 else "inverted"
        report.statuses[center.atom] = status
        report.volumes[center.atom] = volume
    return report


def assert_stereo_preserved(
    target: Union[Primitive, ArrayLike],
    reference: StereoReference,
    box_lengths: Optional[Sequence[float]] = None,
    planar_tolerance: float = DEFAULT_PLANAR_TOLERANCE,
) -> StereoReport:
    """Like `check_stereo`, but raise StereoInversionError unless every center passed"""
    report = check_stereo(target, reference, box_lengths, planar_tolerance)
    if not report.passed:
        raise StereoInversionError(report)
    return report
