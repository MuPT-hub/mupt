"""
Linear chains: the shared chain generator, and generators for linear and ladder polymers

Every chain built here is topologically linear -- an unbranched path of repeat units --
including ladder polymers, whose junctions have several bonds. So one shared, private
generator builds them all, and one thin public wrappers choose what to expose:
    LinearPolymerGenerator : units joined by single bonds
"""

from types import MappingProxyType
from typing import Mapping, Optional
import warnings

import numpy as np
from networkx import Graph

from ...graphs.generators import path_graph_from_sequence
from ...mupr.primitives import Primitive

from .._common import NODE_UNIT, Bond, FaceSpec, GeneratedKey, Node, PortLabel, Site, UnitKey, UnitLabel, check_label
from ..exceptions import FaceError
from ..models.polydispersity import SizeDistribution, coerce_size
from ..models.sequence import SequenceRandom, SequenceModel
from ..models.tacticity import TacticityModel, gaps_between
from ..rules.faces import face_anchor_indices, free_ports_by_label, normalize_faces
from ..rules.junctions import FixedPairing, JunctionRule, anchors_symmetry_equivalent
from ..utils.lexicon import CAP_FACE, CAP_KEY_LABELS, CapSide, UnitLike, build_cap_template, build_unit_templates, cap_key
from .base import TopologyGenerator


# Chains: linear and ladder polymers
_CHAIN_FACES = ("head", "tail")

type EndGroup = UnitLike | Mapping[str, UnitLike]  # a cap, or {label: cap} to name it

#region ABC
class _ChainTopologyGenerator(TopologyGenerator):
    """
    Shared, private base for generators of unbranched chains (linear and ladder polymers)

    Each chain is a path of repeat units joined tail face to head face, with node 0
    at the head end, optionally capped at either end. It holds all of the generation
    logic; the public wrappers (LinearPolymerGenerator, LadderPolymerGenerator) only
    check the junction width they support and choose which arguments to expose, and
    document those arguments for users.
    """

    def __init__(
        self,
        units: Mapping[UnitLabel, UnitLike],
        *,
        size: SizeDistribution | int,
        faces: Optional[Mapping[UnitLabel, Mapping]] = None,
        end_groups: Optional[EndGroup | Mapping[str, Optional[EndGroup]]] = None,
        sequence: Optional[SequenceModel] = None,
        tacticity: Optional[TacticityModel] = None,
        side_groups: Optional[Mapping[UnitLabel, int]] = None,
        junctions: Optional[JunctionRule] = None,
        embed_random_seed: int = 0,
    ) -> None:
        # 1) units and their faces, which must all share one junction width
        if not units:
            raise ValueError("At least one repeat unit must be given")
        for label in units:
            check_label(label, "Unit")

        end_groups = self._normalize_end_groups(end_groups)
        cap_labels = {label for label, _ in end_groups.values() if label is not None}

        faces = faces or {}
        if unknown := set(faces) - set(units) - cap_labels:
            raise FaceError(f"Faces given for unknown unit(s) {sorted(map(str, unknown))}")
        side_groups = side_groups or {}
        if unknown := set(side_groups) - set(units):
            raise ValueError(f"Side groups given for unknown unit(s) {sorted(map(str, unknown))}")

        # faces of the repeat units only; caps' faces are added to the public faces in step 6
        self._unit_faces: dict[UnitLabel, FaceSpec] = {label: normalize_faces(faces.get(label)) for label in units}
        self.width: int = self._shared_width(self._unit_faces)

        # 2) the subclass decides which widths it supports
        self._check_width(self.width)

        # 3, 4) statistics, with defaults
        self.size: SizeDistribution = coerce_size(size)
        self.sequence: SequenceModel = sequence or SequenceRandom({label: 1.0 for label in units})
        self.tacticity: Optional[TacticityModel] = tacticity
        self.junctions: JunctionRule = junctions or FixedPairing()
        if unknown := self.sequence.labels - set(units):
            raise ValueError(f"Sequence model can emit unit(s) {sorted(map(str, unknown))} not given in units")

        # 5) the lexicon: one template per unit, or two stereo variants per tactic unit
        self._lexicon: dict[GeneratedKey, Primitive] = {}
        for label, unit in units.items():
            self._lexicon.update(build_unit_templates(
                label,
                unit,
                self._unit_faces[label],
                tactic=(tacticity is not None),
                side_group=side_groups.get(label),
                required_faces=_CHAIN_FACES,
                embed_random_seed=embed_random_seed,
            ))

        self.tactic_labels: frozenset[UnitLabel] = frozenset(
            label for (label, config) in self._lexicon if config is not None
        )
        if (tacticity is not None) and not self.tactic_labels:
            warnings.warn("A tacticity model was given, but no unit has backbone stereocenters")

        # 6) end groups, added to the lexicon under reserved cap keys (or their own labels, if named)
        self._caps: dict[CapSide, tuple[GeneratedKey, tuple[PortLabel, ...]]] = {}  # side -> (key, 'cap' ports)
        self._cap_faces: dict[UnitLabel, FaceSpec] = {}
        self._build_caps(end_groups, faces, set(units), embed_random_seed)

        # 7) let the junction rule check the faces it will pair up
        self.junctions.validate(self._unit_faces)

    # Construction helpers
    def _check_width(self, width: int) -> None:
        """Raise if this generator can't build chains of the given junction width (default: any)"""

    @staticmethod
    def _shared_width(faces: Mapping[UnitLabel, FaceSpec]) -> int:
        """The number of bonds per junction, which every unit's head and tail faces must share"""
        widths: dict[UnitLabel, tuple[int, int]] = {}
        for label, spec in faces.items():
            if missing := [name for name in _CHAIN_FACES if name not in spec]:
                raise FaceError(f"Unit {label!r} is missing face(s) {missing}; chain units need 'head' and 'tail'")
            widths[label] = (len(spec["head"]), len(spec["tail"]))

        if len({width for pair in widths.values() for width in pair}) != 1:
            raise FaceError(f"Every unit's head and tail faces must have the same width, got (head, tail) = {widths}")

        return next(iter(widths.values()))[0]

    @staticmethod
    def _normalize_end_groups(end_groups) -> dict[CapSide, tuple[Optional[str], UnitLike]]:
        """
        Turn the end_groups argument into {side: (cap label or None, cap)}, for capped sides only

        A single value caps both ends; a {label: cap} value names a cap (None: anonymous)
        """
        if end_groups is None:
            return {}
        if not isinstance(end_groups, Mapping):
            end_groups = {"head": end_groups, "tail": end_groups}  # one cap for both ends
        if unknown := set(end_groups) - set(_CHAIN_FACES):
            raise ValueError(f"End groups must be keyed by 'head' and/or 'tail', got {sorted(map(str, unknown))}")

        normalized: dict[CapSide, tuple[Optional[str], UnitLike]] = {}
        for side in _CHAIN_FACES:
            if (cap := end_groups.get(side)) is None:
                continue  # this end stays free

            if isinstance(cap, Mapping):  # a named cap, {label: cap}
                if len(cap) != 1:
                    raise ValueError(f"A named {side} cap must be given as {{label: cap}}, got {len(cap)} entries")
                ((label, cap),) = cap.items()
                check_label(label, "Cap")
                normalized[side] = (label, cap)
            else:
                normalized[side] = (None, cap)

        return normalized

    def _build_caps(self, end_groups, faces, unit_labels: set, embed_random_seed: int) -> None:
        """Build a cap template for each capped end, and check it fits the chain"""
        named: dict[str, UnitLike] = {}  # label -> cap, to check a label shared by both ends

        for side, (label, cap) in end_groups.items():
            if label is not None:
                if (label in unit_labels) or (label in CAP_KEY_LABELS.values()):
                    raise ValueError(f"Cap label {label!r} is already used by a repeat unit or reserved")
                if (label in named) and (named[label] is not cap) and (named[label] != cap):
                    raise ValueError(f"The head and tail caps share label {label!r} but are given different caps")
                named[label] = cap

            template, cap_faces = build_cap_template(
                cap, side, label=label, faces=faces.get(label) if label else None, embed_random_seed=embed_random_seed,
            )
            ports = cap_faces[CAP_FACE]

            # a cap either covers the whole end face, or is one port copied onto every strand
            if len(ports) not in (1, self.width):
                raise FaceError(
                    f"The {side} cap has {len(ports)} ports; it needs 1 (copied per strand) "
                    f"or {self.width} (one per strand)"
                )
            self._check_cap_bondtypes(side, template, ports)

            key = (label, None) if (label is not None) else cap_key(side)
            self._caps[side] = (key, ports)
            self._cap_faces[key[0]] = cap_faces
            self._lexicon[key] = template

    def _check_cap_bondtypes(self, side: CapSide, template: Primitive, ports: tuple[PortLabel, ...]) -> None:
        """Check a cap's ports have the same bond type as the chain-end ports they'll bond to"""
        cap_types = {free_ports_by_label(template)[port].bondtype for port in ports}
        for (label, _), unit in self._lexicon.items():
            if label not in self._unit_faces:
                continue  # another cap
            unit_ports = free_ports_by_label(unit)
            end_types = {unit_ports[port].bondtype for port in self._unit_faces[label][side]}
            if cap_types != end_types:
                raise FaceError(
                    f"The {side} cap's bond type(s) {cap_types} don't match the {side} "
                    f"face of unit {label!r} ({end_types})"
                )

    # Generation
    @property
    def lexicon(self) -> Mapping[UnitKey, Primitive]:
        return MappingProxyType(self._lexicon)

    @property
    def faces(self) -> Mapping[UnitLabel, FaceSpec]:
        """Faces of every unit label in this generator's graphs: repeat units, then caps"""
        return MappingProxyType({**self._unit_faces, **self._cap_faces})

    def possible_free_sites(self) -> frozenset[Site]:
        """
        The sites that can be left free on a generated chain

        An uncapped end exposes the head (or tail) face of whichever unit lands there;
        a capped end exposes its cap's faces other than 'cap'
        """
        sites: set[Site] = set()
        for side in _CHAIN_FACES:
            if side in self._caps:
                cap_label = self._caps[side][0][0]
                sites.update((cap_label, name) for name in self._cap_faces[cap_label] if name != CAP_FACE)
            else:
                sites.update((label, side) for label in self._unit_faces)

        return frozenset(sites)

    def generate_topology(self, rng: np.random.Generator) -> Graph:
        """
        Build one chain's descriptive graph

        Random numbers are drawn in a fixed order: size, sequence, tacticity, pairings
        """
        n_units = self.size.sample(1, rng)[0]
        labels = self.sequence.sample(n_units, rng)

        # configurations are drawn for the tactic units only, in chain order; the model is
        # told how many non-tactic units separate them (see models.tacticity)
        gaps = gaps_between([label in self.tactic_labels for label in labels])
        configs = iter(self.tacticity.sample(gaps, rng) if self.tacticity else ())
        keys = [(label, next(configs) if (label in self.tactic_labels) else None) for label in labels]

        # a path of units, each one's tail face bonded to the next one's head face
        G = path_graph_from_sequence(keys, label_attr=NODE_UNIT)
        chain_junctions = [(i, "tail", i + 1, "head") for i in range(n_units - 1)]
        self.junctions.assign(G, chain_junctions, self._unit_faces, rng)

        self._attach_caps(G, n_units)
        return G

    def _attach_caps(self, G: Graph, n_units: int) -> None:
        """Add cap nodes at the capped ends of a chain, bonded strand by strand"""
        for side, end_node in (("head", 0), ("tail", n_units - 1)):
            if side not in self._caps:
                continue

            key, cap_ports = self._caps[side]
            end_label = G.nodes[end_node][NODE_UNIT][0]
            end_ports = self._unit_faces[end_label][side]

            # one cap node covering every strand, or one single-port copy per strand
            if len(cap_ports) == len(end_ports):
                strand_groups = [list(zip(cap_ports, end_ports))]
            else:
                strand_groups = [[(cap_ports[0], end_port)] for end_port in end_ports]

            for strands in strand_groups:
                cap_node: Node = G.number_of_nodes()  # next unused integer id
                G.add_node(cap_node, **{NODE_UNIT: key})

                # write bonds upstream-first: head cap -> chain -> tail cap
                bonds: list[Bond] = []
                for cap_port, end_port in strands:
                    cap_end, chain_end = (cap_node, cap_port), (end_node, end_port)
                    bonds.append((cap_end, chain_end) if (side == "head") else (chain_end, cap_end))
                G.add_edge(end_node, cap_node, bonds=tuple(bonds))

    def __repr__(self) -> str:
        units = sorted(self._unit_faces)
        return (
            f"{type(self).__name__}(units={units}, width={self.width}, size={self.size!r}, "
            f"sequence={self.sequence!r}, tacticity={self.tacticity!r}, junctions={self.junctions!r})"
        )


#region Linear polymers
class LinearPolymerGenerator(_ChainTopologyGenerator):
    """
    Generator of linear polymer chains, whose units are joined by single bonds

    Each chain is a path of repeat units, every unit's tail port bonded to the next
    unit's head port, with node 0 at the head end. For chains joined by several bonds
    per junction (ladder polymers), use LadderPolymerGenerator.

    Parameters
    ----------
    units : Mapping[UnitLabel, str | Primitive]
        Repeat units, as SMILES or Primitives, with ports labelled by atom map numbers;
        by default the head port is labelled 1 and the tail port 2, e.g. "*[CH2:1][CH:2](C)*"
    size : SizeDistribution | int
        Number of repeat units per chain (end groups excluded); an int means every
        chain has exactly that many
    faces : Mapping[UnitLabel, {'head': port, 'tail': port}], optional
        Head and tail port labels of units which don't follow the 1/2 convention;
        may also give a named cap's faces (see end_groups)
    end_groups : str | Primitive | {'head': ..., 'tail': ...}, optional
        Caps for the chain ends; a single value caps both ends, None leaves ends free.
        A cap given as {label: cap} is named: it's keyed (label, None), and its faces may
        be given in faces, where the face 'cap' bonds the chain and any other face stays
        free (e.g. {'ester': {'cap': 2, 'link': 5}} for "[*:5]C(=O)[*:2]"); without faces,
        all of a cap's ports bond the chain
    sequence : SequenceModel, optional
        Unit sequence statistics; defaults to an equimolar SequenceRandom
    tacticity : TacticityModel, optional
        Configurations of units with backbone stereocenters; None leaves units as given
    side_groups : Mapping[UnitLabel, int], optional
        Atom map number of the side group used to define a unit's configuration; only
        needed to override the default (highest CIP rank), e.g. to pair styrene's phenyl
        with methyl methacrylate's methyl rather than its ester
    embed_random_seed : int, default 0
        Seed for embedding unit coordinates (-1 for non-deterministic)
    """

    def __init__(
        self,
        units: Mapping[UnitLabel, UnitLike],
        *,
        size: SizeDistribution | int,
        faces: Optional[Mapping[UnitLabel, Mapping]] = None,
        end_groups: Optional[EndGroup | Mapping[str, Optional[EndGroup]]] = None,
        sequence: Optional[SequenceModel] = None,
        tacticity: Optional[TacticityModel] = None,
        side_groups: Optional[Mapping[UnitLabel, int]] = None,
        embed_random_seed: int = 0,
    ) -> None:
        super().__init__(
            units,
            size=size,
            faces=faces,
            end_groups=end_groups,
            sequence=sequence,
            tacticity=tacticity,
            side_groups=side_groups,
            junctions=FixedPairing(),  # one bond per junction: nothing to pair up
            embed_random_seed=embed_random_seed,
        )

    def _check_width(self, width: int) -> None:
        if width != 1:
            raise FaceError(
                f"LinearPolymerGenerator builds chains joined by single bonds, but the faces "
                f"give {width} bonds per junction; use LadderPolymerGenerator instead"
            )


