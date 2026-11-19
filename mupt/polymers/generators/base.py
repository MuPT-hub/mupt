"""
The interface shared by every topology generator

A topology generator is a complete recipe for one kind of polymer -- its repeat units,
sequence and stereo statistics, junctions, end groups, and chain size. It produces
descriptive graphs (see mupt.polymers._common), one per call to generate_topology(),
and realizes them as MuPT Primitives through mupt.polymers.utils.realize.

Realized units carry internal coordinates but are NOT placed relative to one
another; positioning chains is left to a PlacementGenerator.

Chain generators live in linear.py and ladder.py; this module holds only what any
generator shares, plus the base class for composites (generators built from others).
"""

from typing import Hashable, Mapping, Optional
from abc import ABC, abstractmethod
from types import MappingProxyType
import copy
import warnings

import numpy as np
from networkx import Graph

from ...mupr.primitives import Primitive, RootPrimitive

from .._common import ComponentLabel, FaceSpec, Node, SeedLike, Site, UnitKey, UnitLabel, as_rng, check_label
from ..rules.compatibility import CompatibilityRule, head_to_tail
from ..rules.junctions import FixedPairing, JunctionRule
from ..utils.compose import (
    flatten_component_graph,
    generate_node_topology,
    namespace_faces,
    namespace_lexicon,
    resolve_component_graph,
    validate_component_graph,
)
from ..utils.realize import combine_universes, universe_from_descriptive_graphs


# Every topology generator
class TopologyGenerator(ABC):
    """
    Abstract base class for polymer topology generators

    Subclasses implement generate_topology(), which builds ONE descriptive graph, the
    lexicon property, which holds every template those graphs refer to, and the faces
    property, which says how each of those units can bond. Batches and realization
    are handled here.
    """

    @property
    @abstractmethod
    def lexicon(self) -> Mapping[UnitKey, Primitive]:
        """Every template this generator's graphs refer to, keyed by unit key"""
        ...

    @property
    @abstractmethod
    def faces(self) -> Mapping[UnitLabel, FaceSpec]:
        """Faces of every unit label this generator's graphs use (caps included)"""
        ...

    def possible_free_sites(self) -> Optional[frozenset[Site]]:
        """
        Every (unit label, face) that can be left free on a generated graph, or None if unknown

        Composites use this to catch impossible joins at construction; None defers those
        checks until generation
        """
        return None

    @abstractmethod
    def generate_topology(self, rng: np.random.Generator) -> Graph:
        """
        Build one descriptive graph, drawing any random numbers from rng

        Public so that composite generators can call it on the generators they hold
        """
        ...

    def generate_topologies(self, n: int, seed: SeedLike = None) -> list[Graph]:
        """Build n descriptive graphs; the same seed always gives the same graphs"""
        if n < 0:
            raise ValueError(f"Number of topologies must be non-negative, got {n}")

        rng = as_rng(seed)  # one random stream for the whole batch
        return [self.generate_topology(rng) for _ in range(n)]

    def realize(
        self,
        n: int,
        seed: SeedLike = None,
        universe: Optional[RootPrimitive] = None,
        label: Hashable = "universe",
    ) -> RootPrimitive:
        """
        Generate n topologies and realize them as a new universe

        If a universe is given, returns a NEW universe holding copies of its chains
        followed by the new ones (e.g. for blends); the given universe is unchanged
        """
        generated = universe_from_descriptive_graphs(self.generate_topologies(n, seed), self.lexicon, label=label)
        if universe is None:
            return generated

        return combine_universes(universe, generated, label=label)


# Generators built from other generators
class CompositeGenerator(TopologyGenerator):
    """
    Abstract base class for generators whose topology is a graph of other generators

    The generators a composite is built from are its components, given as a dict
    {label: generator} -- the composite's counterpart of a chain generator's units. A
    composite so has two "lexicons": its components, and the merged unit lexicon of
    those components.

    Subclasses implement component_graph(), which says which components appear and
    which of their faces are fixed at each join (see utils.compose). Merging lexicons,
    namespacing, matching faces, pairing strands and flattening are handled here.

    Unit keys are namespaced by component as flat paths, e.g. (('PS', 'S'), 1). A SMILES
    or Primitive given as a component is wrapped as UnitGenerator(unit), keyed
    ((component, 'unit'), None).

    Each component samples its own configurations, so tacticity is NOT carried across
    joins: diads spanning two components are uncorrelated.
    """

    def __init__(
        self,
        components: Mapping[ComponentLabel, "TopologyGenerator | str | Primitive"],
        *,
        compatibility: Optional[CompatibilityRule] = None,
        junctions: Optional[JunctionRule] = None,
    ) -> None:
        from .unit import UnitGenerator  # unit.py builds on this module

        if not components:
            raise ValueError("At least one component must be given")

        # 1) components, with bare units wrapped as one-unit generators
        self._components: dict[ComponentLabel, TopologyGenerator] = {}
        for label, component in components.items():
            check_label(label, "Component")
            if isinstance(component, (str, Primitive)):
                component = UnitGenerator(component)
            elif not isinstance(component, TopologyGenerator):
                raise TypeError(f"Component {label!r} must be a TopologyGenerator, SMILES or Primitive, got {component!r}")
            self._components[label] = component

        # 2) the merged unit lexicon and faces, namespaced by component
        self._lexicon: dict[UnitKey, Primitive] = {}
        self._faces: dict[UnitLabel, FaceSpec] = {}
        for label, component in self._components.items():
            self._lexicon.update(namespace_lexicon(label, component.lexicon))
            self._faces.update(namespace_faces(label, component.faces))

        # 3) rules, checked against the merged faces (copied, since validating stores them)
        self.compatibility: CompatibilityRule = copy.copy(compatibility or self.default_compatibility())
        self.compatibility.validate(self._faces, set(self._components))
        self.junctions: JunctionRule = junctions or FixedPairing()
        self.junctions.validate(self._faces)

    # The two "lexicons"
    @property
    def components(self) -> Mapping[ComponentLabel, TopologyGenerator]:
        """Every component, by label"""
        return MappingProxyType(self._components)

    @property
    def lexicon(self) -> Mapping[UnitKey, Primitive]:
        """The merged unit lexicon of every component, with namespaced keys"""
        return MappingProxyType(self._lexicon)

    @property
    def faces(self) -> Mapping[UnitLabel, FaceSpec]:
        """The merged faces of every component, with namespaced unit labels"""
        return MappingProxyType(self._faces)

    # What subclasses write
    def default_compatibility(self) -> CompatibilityRule:
        """The compatibility rule used when none is given (default: any free tail to any free head)"""
        return head_to_tail()

    @abstractmethod
    def component_graph(self, rng: np.random.Generator) -> Graph:
        """
        One unresolved component graph: nodes hold 'component', edges hold 'joins'

        May resolve nodes early with resolve_node(), when the shape depends on them
        (e.g. a star's arms depend on its core's free faces)
        """
        ...

    # Generation
    def resolve_node(self, coarse: Graph, node: Node, rng: np.random.Generator) -> Graph:
        """Generate one node's topology now (namespaced), store it on the node, and return it"""
        return generate_node_topology(coarse, node, self._components, rng)

    def generate_component_graph(self, rng: np.random.Generator) -> Graph:
        """One resolved component graph, e.g. to inspect or edit before flattening"""
        coarse = self.component_graph(rng)
        validate_component_graph(coarse, self._components)
        return resolve_component_graph(coarse, self._components, self._faces, self.compatibility, self.junctions, rng)

    def generate_topology(self, rng: np.random.Generator) -> Graph:
        """Build one unit graph: generate the component graph, resolve it, then flatten it"""
        return flatten_component_graph(self.generate_component_graph(rng))

    # Helpers for subclasses
    def _warn_unused(self, used) -> None:
        """Warn about components that a composite's arrangement never uses"""
        if unused := [label for label in self._components if label not in set(used)]:
            warnings.warn(f"Component(s) {unused} are never used")
