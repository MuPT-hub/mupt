"""Shared types, constants, and helpers for polymer topology generation"""

from typing import Hashable, Literal, Mapping, NamedTuple

import numpy as np

# Labels and identifiers
type UnitLabel = Hashable  # name of a repeat unit, e.g. 'styrene'
type PortLabel = Hashable  # Connector.label of a port (int map numbers in practice)
type Config = Literal[1, -1]  # tactic configuration of a unit
type UnitKey = Hashable  # lexicon key; any hashable for hand-built lexicons
type GeneratedKey = tuple[UnitLabel, Config | None]  # the lexicon keys generators produce
type Node = Hashable  # node of a descriptive graph

# Faces: named, ordered groups of ports which all bond to the same neighbor
type FaceName = Hashable
type Face = tuple[PortLabel, ...]  # ordered by strand
type FaceSpec = Mapping[FaceName, Face]  # all faces of one unit

# Junctions and bonds
type Pairing = tuple[int, ...]  # strand k of face_u bonds to strand pairing[k] of face_v
type Junction = tuple[Node, FaceName, Node, FaceName]  # (u, face_u, v, face_v), u upstream
type Bond = tuple[tuple[Node, PortLabel], tuple[Node, PortLabel]]  # orientation-free

# Composites: graphs whose nodes are other generators
type ComponentLabel = str  # name of a component in a composite, e.g. 'PS' or 'arm_A'
type Site = tuple[UnitLabel, FaceName]  # the TYPE of an attachment point: (unit label or path, face name)
type FreeFace = tuple[Node, FaceName]  # one node's face, with none of its ports bonded


class FixedFace(NamedTuple):
    """
    A join anchor naming one free face of an already resolved component, by its node in
    that component's topology; a distinct type, so it's never read as a site pattern
    """
    node: Node
    face: FaceName

type SeedLike = int | np.random.Generator | None


def as_rng(seed: SeedLike) -> np.random.Generator:
    """Coerce a seed (or existing Generator) into a numpy Generator"""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# Descriptive graph schema (the whole contract between generation and realization)
NODE_UNIT = "unit"  # UnitKey of the lexicon template to copy
EDGE_BONDS = "bonds"  # tuple[Bond, ...], one entry per bond between the edge's endpoints

# Component graph schema (composites; see utils.compose)
NODE_COMPONENT = "component"  # ComponentLabel of the generator this node stands for
NODE_TOPOLOGY = "topology"  # once resolved: the component's generated (namespaced) unit graph
EDGE_JOINS = "joins"  # ((u, anchor_u), (v, anchor_v)), upstream first; resolved edges also get EDGE_BONDS

# Metadata keys set on lexicon templates (inherited by every realized copy)
UNIT_LABEL_KEY = "unit_label"
CONFIG_KEY = "config"
RESIDUE_NAME_KEY = "residue_name"  # read by the shared exporter traversal
COMPONENT_KEY = "component"  # composites only: tuple of component labels, outermost first


def check_label(label: object, kind: str) -> None:
    """
    Require a unit, cap, or component label to be a str

    Composites namespace keys as flat tuple paths, e.g. ('PS', 'S'), so a tuple must
    always mean a path and never a label
    """
    if not isinstance(label, str):
        raise TypeError(f"{kind} labels must be str, got {label!r} ({type(label).__name__})")
