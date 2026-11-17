"""
Port compatibility: deciding which free faces bond when components are joined

A site is the TYPE of an attachment point, (unit label, face name). A compatibility rule
says which upstream sites may bond to which downstream sites; matching then picks, among
the free faces of two graphs, the pair that bonds, and a junction rule pairs its strands.

Rules work on faces, so ladder faces match as a whole; users may still name a face by
any one of its port labels. Sites are named with patterns (see resolve_pattern), read
by their length:

    at composite level                      inside one component (e.g. a star's arms)
        x                    any unit           x            any unit
        (component, x)       one component      (unit, x)    one unit
        (component, unit, x) one unit of one component
        (None, unit, x)      one unit, in any component

x is a face name, else the face holding port x; None in any position is a wildcard,
and ANY matches every site.
"""

from typing import Collection, Hashable, Iterable, Literal, Mapping, Optional, Sequence
from abc import ABC, abstractmethod

import numpy as np
from networkx import Graph

from .._common import EDGE_BONDS, NODE_UNIT, FaceName, FaceSpec, FreeFace, Node, Site, UnitLabel
from ..exceptions import CompatibilityError, FaceError
from .junctions import faces_of_node

type IfAmbiguous = Literal["error", "random", "first"]
_IF_AMBIGUOUS = ("error", "random", "first")


class _AnySite:
    """Sentinel pattern matching every site"""

    def __repr__(self) -> str:
        return "ANY"


ANY = _AnySite()


# Reading free faces off a graph (derived on demand, never stored)
def unit_label_of(G: Graph, node: Node, faces: Mapping[UnitLabel, FaceSpec]) -> UnitLabel:
    """
    The unit label a node's faces are keyed by

    Same rule as faces_of_node: the key itself if faces has it, else key[0] of a
    (label, config) key
    """
    key = G.nodes[node][NODE_UNIT]
    if key in faces:
        return key
    if isinstance(key, tuple) and key and (key[0] in faces):
        return key[0]

    raise FaceError(f"No faces given for unit {key!r} of node {node!r}")


def free_faces(G: Graph, faces: Mapping[UnitLabel, FaceSpec]) -> list[FreeFace]:
    """
    Every face with none of its ports used in any edge's bonds, in node order

    A face with only SOME of its ports bonded can't take part in a join, so it raises
    """
    used: dict[Node, set] = {node: set() for node in G}
    for _, _, bonds in G.edges(data=EDGE_BONDS, default=()):
        for (node_a, port_a), (node_b, port_b) in bonds:
            used[node_a].add(port_a)
            used[node_b].add(port_b)

    found: list[FreeFace] = []
    for node in G:
        for name, ports in faces_of_node(G, node, faces).items():
            n_used = sum(port in used[node] for port in ports)
            if n_used == 0:
                found.append((node, name))
            elif n_used < len(ports):
                raise CompatibilityError(
                    f"Face {name!r} of node {node!r} is partly bonded ({n_used} of {len(ports)} ports)"
                )

    return found


def site_of(G: Graph, free_face: FreeFace, faces: Mapping[UnitLabel, FaceSpec]) -> Site:
    """The site (unit label, face name) of a free face"""
    node, name = free_face
    return (unit_label_of(G, node, faces), name)


# Site patterns
def _faces_matching(spec: FaceSpec, x) -> list[FaceName]:
    """The face named x, else the face holding port x, else nothing (wildcard: every face)"""
    if x is None:
        return list(spec)
    if x in spec:
        return [x]
    return [name for name, ports in spec.items() if x in ports]


def _unit_matches(label: UnitLabel, unit) -> bool:
    """Whether a label is the given unit: equal, or for a path, its last element or its rest"""
    if unit is None:
        return True
    if isinstance(label, tuple):
        return (label[-1] == unit) or (isinstance(unit, tuple) and label[1:] == unit)
    return label == unit


def resolve_pattern(
    pattern,
    faces: Mapping[UnitLabel, FaceSpec],
    components: Optional[Collection[Hashable]] = None,
) -> frozenset[Site]:
    """
    Turn a site pattern into the set of sites it names

    With components given, the pattern is read at composite level, where unit labels are
    paths starting with a component label (see module docstring); without, it's read
    inside one component. A pattern naming no site raises
    """
    if pattern is ANY:
        return frozenset((label, name) for label, spec in faces.items() for name in spec)

    # split the pattern into (component, unit, x) by its length
    composite = components is not None
    if not isinstance(pattern, tuple):
        component, unit, x = None, None, pattern
    elif composite and len(pattern) == 2:
        (component, x), unit = pattern, None
    elif composite and len(pattern) == 3:
        component, unit, x = pattern
    elif (not composite) and len(pattern) == 2:
        (unit, x), component = pattern, None
    else:
        forms = "x, (component, x) or (component, unit, x)" if composite else "x or (unit, x)"
        raise CompatibilityError(f"Can't read site pattern {pattern!r}; expected {forms}")

    if (component is not None) and (component not in components):
        raise CompatibilityError(
            f"Site pattern {pattern!r} names component {component!r}, which isn't one of "
            f"{sorted(map(str, components))} (to name a unit in any component, use (None, unit, x))"
        )

    sites = set()
    for label, spec in faces.items():
        if (component is not None) and not (isinstance(label, tuple) and label[0] == component):
            continue
        if not _unit_matches(label, unit):
            continue
        sites.update((label, name) for name in _faces_matching(spec, x))

    if not sites:
        units = sorted({str(label) for label in faces})
        raise CompatibilityError(f"Site pattern {pattern!r} matches no face of any unit in scope: {units}")

    return frozenset(sites)


# The framework
class CompatibilityRule(ABC):
    """
    Abstract base class for which free faces bond when two components are joined

    Rules are DIRECTIONAL: compatible(upstream, downstream). Without direction, a
    block's tail-to-head join would also fit head-to-tail.

    if_ambiguous decides what happens when several pairs of free faces fit one join:
        'error'  : raise, naming the competing sites (default)
        'random' : pick one uniformly; exactly one random number is drawn per match,
                   however many pairs fit, so ties never shift the random stream
        'first'  : take the first pair in node order (upstream first)
    """

    def __init__(self, *, if_ambiguous: IfAmbiguous = "error") -> None:
        if if_ambiguous not in _IF_AMBIGUOUS:
            raise ValueError(f"if_ambiguous must be one of {_IF_AMBIGUOUS}, got {if_ambiguous!r}")
        self.if_ambiguous = if_ambiguous
        self._faces: Mapping[UnitLabel, FaceSpec] = {}

    def validate(self, faces: Mapping[UnitLabel, FaceSpec], components: Optional[Collection[Hashable]] = None) -> None:
        """Check the rule against every unit's faces, when a generator is built (default: just keep them)"""
        self._faces = faces

    @abstractmethod
    def compatible(self, upstream: Site, downstream: Site) -> bool:
        """Whether an upstream site may bond to a downstream site (static: no graph, no randomness)"""
        ...

    def width(self, site: Site) -> int:
        """Number of ports in a site's face"""
        label, name = site
        return len(self._faces[label][name])

    def candidates(self, site: Site, sites: Iterable[Site], *, upstream: bool = True) -> list[Site]:
        """The sites that may bond to a given site, which is upstream unless told otherwise"""
        if upstream:
            return [other for other in sites if self.compatible(site, other)]
        return [other for other in sites if self.compatible(other, site)]

    def match(
        self,
        G_u: Graph,
        free_u: Sequence[FreeFace],
        G_v: Graph,
        free_v: Sequence[FreeFace],
        rng: np.random.Generator,
    ) -> tuple[FreeFace, FreeFace]:
        """
        The pair of free faces, one upstream and one downstream, that bond at a join

        Give a single free face on one side to anchor it (e.g. a star's core face), or
        every free face on both sides to infer the join (e.g. between two blocks)
        """
        pairs = [
            (a, b)
            for a in free_u
            for b in free_v
            if self.compatible(site_of(G_u, a, self._faces), site_of(G_v, b, self._faces))
        ]
        if not pairs:
            up = sorted({str(site_of(G_u, a, self._faces)) for a in free_u})
            down = sorted({str(site_of(G_v, b, self._faces)) for b in free_v})
            raise CompatibilityError(f"No compatible faces to join: upstream {up}, downstream {down}")

        return self.choose(pairs, rng, G_u, G_v)

    def choose(self, pairs, rng: np.random.Generator, G_u: Graph, G_v: Graph) -> tuple[FreeFace, FreeFace]:
        """Pick one of several compatible pairs, as if_ambiguous says"""
        if self.if_ambiguous == "random":
            draw = rng.random()  # one draw per match, even when only one pair fits
            return pairs[min(int(draw * len(pairs)), len(pairs) - 1)]
        if (len(pairs) == 1) or (self.if_ambiguous == "first"):
            return pairs[0]

        described = [(site_of(G_u, a, self._faces), site_of(G_v, b, self._faces)) for a, b in pairs]
        raise CompatibilityError(
            f"{len(pairs)} pairs of faces fit this join: {described}; use a more specific rule, "
            f"or if_ambiguous='random' or 'first'"
        )


# Implementations
class CompatibleSites(CompatibilityRule):
    """
    A table of (upstream pattern, downstream pattern) pairs that may bond

    e.g. CompatibleSites(
            (("hub", 1), ("arm_A", "link")),
            (("hub", 4), ("arm_B", "S", "head")),
        )

    Only faces of equal width are compatible. symmetric=True also lets each pair bond
    the other way round.
    """

    def __init__(self, *pairs, symmetric: bool = False, if_ambiguous: IfAmbiguous = "error") -> None:
        super().__init__(if_ambiguous=if_ambiguous)
        if not pairs:
            raise ValueError("CompatibleSites needs at least one (upstream, downstream) pair")
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(f"Each entry must be an (upstream, downstream) pattern pair, got {pair!r}")

        self.pairs = pairs
        self.symmetric = symmetric
        self._table: Optional[frozenset[tuple[Site, Site]]] = None

    def validate(self, faces, components=None) -> None:
        super().validate(faces, components)

        table: set[tuple[Site, Site]] = set()
        for up_pattern, down_pattern in self.pairs:
            ups = resolve_pattern(up_pattern, faces, components)
            downs = resolve_pattern(down_pattern, faces, components)

            fitting = {(up, down) for up in ups for down in downs if self.width(up) == self.width(down)}
            if not fitting:
                raise CompatibilityError(
                    f"No faces of equal width in {up_pattern!r} -> {down_pattern!r}"
                )
            table.update(fitting)
            if self.symmetric:
                table.update((down, up) for up, down in fitting)

        self._table = frozenset(table)

    def compatible(self, upstream: Site, downstream: Site) -> bool:
        if self._table is None:
            raise RuntimeError("CompatibleSites must be validated against a generator's faces before use")
        return (upstream, downstream) in self._table

    def __repr__(self) -> str:
        pairs = ", ".join(f"({up!r}, {down!r})" for up, down in self.pairs)
        return f"CompatibleSites({pairs}, symmetric={self.symmetric}, if_ambiguous={self.if_ambiguous!r})"


def head_to_tail(**kwargs) -> CompatibleSites:
    """Any unit's free tail bonds to any unit's free head: the default for block joins"""
    return CompatibleSites(((None, None, "tail"), (None, None, "head")), **kwargs)
