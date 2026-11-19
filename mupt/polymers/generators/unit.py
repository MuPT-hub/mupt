"""
Single units as components, e.g. the core of a star polymer
"""

from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np
from networkx import Graph

from ...mupr.primitives import Primitive

from .._common import NODE_UNIT, FaceSpec, GeneratedKey, Site, UnitKey, UnitLabel, check_label
from ..rules.faces import free_ports_by_label, normalize_faces, port_atom_indices
from ..utils.lexicon import UnitLike, build_unit_templates, unit_mol
from .base import TopologyGenerator


class UnitGenerator(TopologyGenerator):
    """
    A single unit as a component, e.g. a star's core

    Every graph is one node, keyed (label, None), and generating draws no random numbers.

    A SMILES or Primitive given directly in a composite's components is wrapped as
    UnitGenerator(unit) with the default label 'unit', so its key in that composite is
    ((component label, 'unit'), None), e.g. (('hub', 'unit'), None)

    Parameters
    ----------
    unit : str | Primitive
        The unit, with ports labelled by atom map numbers
    label : str, default 'unit'
        The unit's label in this generator's lexicon
    faces : FaceSpec, optional
        The unit's faces; by default each free port is its own width-1 face, named by
        its port label (e.g. {1: (1,), 2: (2,), ...})
    embed_random_seed : int, default 0
        Seed for embedding the unit's coordinates (-1 for non-deterministic)
    """

    def __init__(
        self,
        unit: UnitLike,
        *,
        label: str = "unit",
        faces: Optional[Mapping] = None,
        embed_random_seed: int = 0,
    ) -> None:
        check_label(label, "Unit")
        self.label = label

        if faces is None:
            # one face per port, named by its label
            ports = free_ports_by_label(unit) if isinstance(unit, Primitive) else port_atom_indices(unit_mol(unit))
            faces = {port: (port,) for port in ports}
        self._faces: dict[UnitLabel, FaceSpec] = {label: normalize_faces(faces)}

        self._lexicon: dict[GeneratedKey, Primitive] = build_unit_templates(
            label, unit, self._faces[label], embed_random_seed=embed_random_seed,
        )

    @property
    def lexicon(self) -> Mapping[UnitKey, Primitive]:
        return MappingProxyType(self._lexicon)

    @property
    def faces(self) -> Mapping[UnitLabel, FaceSpec]:
        return MappingProxyType(self._faces)

    def possible_free_sites(self) -> frozenset[Site]:
        """Every face: nothing on a lone unit is bonded"""
        return frozenset((self.label, name) for name in self._faces[self.label])

    def generate_topology(self, rng: np.random.Generator) -> Graph:
        G = Graph()
        G.add_node(0, **{NODE_UNIT: (self.label, None)})
        return G

    def __repr__(self) -> str:
        return f"UnitGenerator(label={self.label!r}, faces={self._faces[self.label]!r})"
