"""
Star polymers: arms bonded to the free faces of a core
"""

from typing import Mapping, Optional

import numpy as np
from networkx import Graph

from ...mupr.primitives import Primitive

from .._common import EDGE_JOINS, NODE_COMPONENT, ComponentLabel, FixedFace, Site
from ..exceptions import CompatibilityError
from ..rules.compatibility import ANY, CompatibilityRule, CompatibleSites, free_faces, resolve_pattern, site_of
from ..rules.junctions import JunctionRule
from ..utils.compose import namespace_label
from .base import CompositeGenerator, TopologyGenerator
from .unit import UnitGenerator


class StarPolymerGenerator(CompositeGenerator):
    """
    Generator of star polymers: one arm on each chosen free face of a core

    The core is a component like any other: a SMILES (wrapped as a single unit) or any
    generator. Each key of arms names faces of the core with site patterns, read in the
    core's own scope (x: a face name or port label; (unit, x): on one unit of the core),
    and EVERY free face of the generated core matching a key gets its own arm. A tuple
    key holds several patterns, e.g. (1, 2, 3); to use a (unit, x) pattern, put it in a
    tuple of its own, e.g. (('S', 9),).

    Each arm is generated independently, in core-face order, so arms have their own
    sizes and tacticity. The compatibility rule picks which free face of each arm bonds
    to its core face (by default, the arm's free 'head').

    Parameters
    ----------
    components : Mapping[str, TopologyGenerator | str | Primitive]
        The core and arm generators, by label, e.g.
        {'hub': '[*:1]...[*:6]', 'arm_A': arm_generator_1, 'arm_B': arm_generator_2}
    core : str
        Label of the core component
    arms : Mapping[pattern or tuple of patterns, str]
        Core faces -> arm component label, e.g. {(1, 2, 3): 'arm_A', (4, 5, 6): 'arm_B'}
    compatibility : CompatibilityRule, optional
        Which free face of each arm bonds to its core face; defaults to
        CompatibleSites((ANY, (None, None, 'head')))
    junctions : JunctionRule, optional
        How strands pair at each core-arm bond (for ladder arms); defaults to FixedPairing
    allow_free_core_ports : bool, default False
        Whether free faces of the core may be left without an arm; if not, they raise
    """

    def __init__(
        self,
        components: Mapping[ComponentLabel, TopologyGenerator | str | Primitive],
        core: ComponentLabel,
        arms: Mapping[object, ComponentLabel],
        *,
        compatibility: Optional[CompatibilityRule] = None,
        junctions: Optional[JunctionRule] = None,
        allow_free_core_ports: bool = False,
    ) -> None:
        super().__init__(components, compatibility=compatibility, junctions=junctions)

        if core not in self._components:
            raise ValueError(f"Core {core!r} isn't among the components {list(self._components)}")
        if not arms:
            raise ValueError("At least one arm must be given")
        if unknown := sorted({label for label in arms.values() if label not in self._components}):
            raise ValueError(f"Arm(s) {unknown} aren't among the components {list(self._components)}")

        self.core = core
        self.arms = dict(arms)
        self.allow_free_core_ports = allow_free_core_ports
        self._warn_unused([core, *self.arms.values()])

        # each key -> the (namespaced) core sites it names
        core_faces = self._components[core].faces
        self._arm_sites: dict[object, frozenset[Site]] = {}
        for key in self.arms:
            patterns = key if isinstance(key, tuple) else (key,)
            sites = set()
            for pattern in patterns:
                sites.update(resolve_pattern(pattern, core_faces))
            self._arm_sites[key] = frozenset((namespace_label(core, label), face) for label, face in sites)

        self._check_arms_can_join()
        if isinstance(self._components[core], UnitGenerator):
            self._check_core_coverage()  # a lone unit's free faces are known now

    def default_compatibility(self) -> CompatibilityRule:
        """Any core face bonds to the arm's free head"""
        return CompatibleSites((ANY, (None, None, "head")))

    # Construction checks
    def _check_arms_can_join(self) -> None:
        """Raise if some arm has no free site compatible with the core faces it's meant for"""
        for key, label in self.arms.items():
            arm_sites = self._components[label].possible_free_sites()
            if arm_sites is None:
                continue  # unknown until generation

            downs = [(namespace_label(label, unit), face) for unit, face in arm_sites]
            if not any(self.compatibility.compatible(up, down) for up in self._arm_sites[key] for down in downs):
                raise CompatibilityError(
                    f"Arm {label!r} has no free site compatible with core faces {key!r} "
                    f"(its free sites: {sorted(map(str, arm_sites))})"
                )

    def _check_core_coverage(self) -> None:
        """For a single-unit core: every free face gets exactly one arm, unless free faces are allowed"""
        core_sites = {(namespace_label(self.core, label), face) for label, face in self._components[self.core].possible_free_sites()}
        self._arms_for_sites(core_sites)

    def _arms_for_sites(self, sites) -> dict[Site, ComponentLabel]:
        """Which arm each core site gets; raises on a site claimed twice, or left free when not allowed"""
        assigned: dict[Site, ComponentLabel] = {}
        unfilled = []
        for site in sites:
            keys = [key for key, claimed in self._arm_sites.items() if site in claimed]
            if len(keys) > 1:
                raise CompatibilityError(f"Core face {site!r} is matched by several arms keys: {keys}")
            if keys:
                assigned[site] = self.arms[keys[0]]
            else:
                unfilled.append(site)

        if unfilled and not self.allow_free_core_ports:
            raise CompatibilityError(
                f"Core face(s) {sorted(map(str, unfilled))} have no arm; give them one, "
                f"or pass allow_free_core_ports=True"
            )
        return assigned

    # Generation
    def component_graph(self, rng: np.random.Generator) -> Graph:
        """The core (resolved now, since its free faces decide the arms), then one arm per matched face"""
        coarse = Graph()
        coarse.add_node(0, **{NODE_COMPONENT: self.core})
        topology = self.resolve_node(coarse, 0, rng)

        core_free = free_faces(topology, self._faces)
        assigned = self._arms_for_sites({site_of(topology, face, self._faces) for face in core_free})

        for unit_node, face in core_free:
            site = site_of(topology, (unit_node, face), self._faces)
            if site not in assigned:
                continue  # left free (allowed)

            arm = coarse.number_of_nodes()
            coarse.add_node(arm, **{NODE_COMPONENT: assigned[site]})
            coarse.add_edge(0, arm, **{EDGE_JOINS: ((0, FixedFace(unit_node, face)), (arm, None))})

        return coarse

    def __repr__(self) -> str:
        return f"StarPolymerGenerator(core={self.core!r}, arms={self.arms!r}, compatibility={self.compatibility!r})"
