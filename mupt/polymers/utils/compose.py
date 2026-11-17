"""
Composition: building one descriptive graph out of the graphs of several components

A composite generator describes its shape as a component graph, whose nodes are
components (other generators) instead of units:

    node  'component' : the component's label
    edge  'joins'     : ((u, anchor_u), (v, anchor_v)), upstream first; each anchor is
                        None (any free face), a site pattern in that component's scope,
                        or a FixedFace of an already resolved node

Resolving it generates each component's unit graph ('topology', with namespaced keys),
matches the faces at each join, and pairs their strands, writing 'bonds' whose nodes are
addresses (coarse node, unit node). Flattening then relabels everything into one plain
unit graph.

Keys are namespaced by component as flat tuple paths: ('S', 1) in component 'PS'
becomes (('PS', 'S'), 1), and (('star', 'PS', 'S'), 1) if that composite is nested again.
"""

from typing import Mapping

import numpy as np
from networkx import Graph

from ...mupr.primitives import Primitive

from .._common import (
    COMPONENT_KEY,
    EDGE_BONDS,
    EDGE_JOINS,
    NODE_COMPONENT,
    NODE_TOPOLOGY,
    NODE_UNIT,
    ComponentLabel,
    FaceSpec,
    FixedFace,
    FreeFace,
    GeneratedKey,
    Junction,
    Node,
    UnitKey,
    UnitLabel,
)
from ..exceptions import CompatibilityError, TopologyValidationError
from ..rules.compatibility import CompatibilityRule, free_faces, resolve_pattern
from ..rules.junctions import JunctionRule

type Address = tuple[Node, Node]  # (coarse node, unit node in that node's topology)


# Namespacing
def namespace_label(name: ComponentLabel, label: UnitLabel) -> UnitLabel:
    """Prefix a unit label with a component label, keeping paths flat: 'S' -> ('PS', 'S')"""
    if isinstance(label, tuple):
        return (name, *label)  # already a path (labels themselves are always str)
    return (name, label)


def namespace_key(name: ComponentLabel, key: GeneratedKey) -> GeneratedKey:
    """Prefix the label of a (label, config) key: ('S', 1) -> (('PS', 'S'), 1)"""
    if not (isinstance(key, tuple) and len(key) == 2):
        raise TypeError(f"Only (label, config) keys can be namespaced, got {key!r}")
    label, config = key
    return (namespace_label(name, label), config)


def namespace_lexicon(name: ComponentLabel, lexicon: Mapping[UnitKey, Primitive]) -> dict[UnitKey, Primitive]:
    """
    Copies of a lexicon's templates under namespaced keys

    Each copy records which components it sits in (outermost first) as 'component'
    metadata; its unit_label, config and residue_name are unchanged
    """
    namespaced: dict[UnitKey, Primitive] = {}
    for key, template in lexicon.items():
        copy = template.copy()
        copy.metadata[COMPONENT_KEY] = (name, *template.metadata.get(COMPONENT_KEY, ()))
        namespaced[namespace_key(name, key)] = copy

    return namespaced


def namespace_faces(name: ComponentLabel, faces: Mapping[UnitLabel, FaceSpec]) -> dict[UnitLabel, FaceSpec]:
    """A faces mapping with every unit label namespaced"""
    return {namespace_label(name, label): spec for label, spec in faces.items()}


def namespace_graph(G: Graph, name: ComponentLabel) -> Graph:
    """A copy of a unit graph with every node's unit key namespaced"""
    H = G.copy()
    for node, key in G.nodes(data=NODE_UNIT):
        H.nodes[node][NODE_UNIT] = namespace_key(name, key)
    return H


# Resolving a component graph
def generate_node_topology(coarse: Graph, node: Node, components: Mapping[ComponentLabel, object], rng) -> Graph:
    """Generate one coarse node's unit graph, namespace it by its component, and store it on the node"""
    label = coarse.nodes[node][NODE_COMPONENT]
    if label not in components:
        raise TopologyValidationError(f"Node {node!r} names unknown component {label!r}")

    topology = namespace_graph(components[label].generate_topology(rng), label)
    coarse.nodes[node][NODE_TOPOLOGY] = topology
    return topology


def _candidates(coarse: Graph, node: Node, anchor, free: list[FreeFace], faces) -> list[FreeFace]:
    """The free faces of one join side that its anchor allows"""
    if anchor is None:
        return list(free)

    if isinstance(anchor, FixedFace):
        if (anchor.node, anchor.face) not in free:
            raise CompatibilityError(f"Face {anchor.face!r} of unit {anchor.node!r} in node {node!r} isn't free")
        return [(anchor.node, anchor.face)]

    # a site pattern, read inside this node's component
    component = coarse.nodes[node][NODE_COMPONENT]
    scope = {label: spec for label, spec in faces.items() if label[0] == component}
    allowed = resolve_pattern(anchor, scope)
    topology = coarse.nodes[node][NODE_TOPOLOGY]
    return [
        (unit_node, face) for unit_node, face in free
        if (_label_of(topology, unit_node), face) in allowed
    ]


def _label_of(topology: Graph, unit_node: Node) -> UnitLabel:
    """The (namespaced) unit label of a node in a resolved topology"""
    return topology.nodes[unit_node][NODE_UNIT][0]


def resolve_component_graph(
    coarse: Graph,
    components: Mapping[ComponentLabel, object],
    faces: Mapping[UnitLabel, FaceSpec],
    compatibility: CompatibilityRule,
    junctions: JunctionRule,
    rng: np.random.Generator,
) -> Graph:
    """
    A resolved copy of a component graph

    Random numbers are drawn in a fixed order:
        1) each node without a topology yet generates one, in node order
        2) each join's faces are matched, in edge order (draws only if the rule chooses randomly)
        3) every join's strands are paired, in one call to the junction rule
    A free face takes part in one join at most. faces must be the merged, namespaced faces
    of every component, and the rules must already be validated against them
    """
    resolved = coarse.copy()

    # 1) every component's unit graph
    for node in resolved:
        if NODE_TOPOLOGY not in resolved.nodes[node]:
            generate_node_topology(resolved, node, components, rng)

    # 2) which faces meet at each join
    free = {node: free_faces(topology, faces) for node, topology in resolved.nodes(data=NODE_TOPOLOGY)}
    edge_junctions: list[tuple[tuple[Node, Node], Junction]] = []

    for edge_u, edge_v, joins in resolved.edges(data=EDGE_JOINS):
        (u, anchor_u), (v, anchor_v) = _unpack_joins(joins, (edge_u, edge_v))
        a, b = compatibility.match(
            resolved.nodes[u][NODE_TOPOLOGY], _candidates(resolved, u, anchor_u, free[u], faces),
            resolved.nodes[v][NODE_TOPOLOGY], _candidates(resolved, v, anchor_v, free[v], faces),
            rng,
        )
        free[u].remove(a)
        free[v].remove(b)
        edge_junctions.append(((edge_u, edge_v), ((u, a[0]), a[1], (v, b[0]), b[1])))

    # 3) strand pairing, run on a small graph whose nodes are the joined units' addresses
    endpoints = Graph()
    for _, (addr_u, _, addr_v, _) in edge_junctions:
        for coarse_node, unit_node in (addr_u, addr_v):
            unit_key = resolved.nodes[coarse_node][NODE_TOPOLOGY].nodes[unit_node][NODE_UNIT]
            endpoints.add_node((coarse_node, unit_node), **{NODE_UNIT: unit_key})
        endpoints.add_edge(addr_u, addr_v)

    junctions.assign(endpoints, [junction for _, junction in edge_junctions], faces, rng)
    for (edge_u, edge_v), (addr_u, _, addr_v, _) in edge_junctions:
        resolved.edges[edge_u, edge_v][EDGE_BONDS] = endpoints.edges[addr_u, addr_v][EDGE_BONDS]

    return resolved


def _unpack_joins(joins, edge) -> tuple[tuple[Node, object], tuple[Node, object]]:
    """Check one edge's joins record and return ((u, anchor_u), (v, anchor_v))"""
    try:
        (u, anchor_u), (v, anchor_v) = joins
    except (TypeError, ValueError):
        raise TopologyValidationError(f"Edge {edge!r} needs joins ((u, anchor), (v, anchor)), got {joins!r}") from None
    if {u, v} != set(edge) or (u == v):
        raise TopologyValidationError(f"Joins {joins!r} don't match edge {edge!r}")
    return (u, anchor_u), (v, anchor_v)


# Flattening
def flatten_component_graph(coarse: Graph) -> Graph:
    """
    One unit graph from a resolved component graph (no random numbers)

    Nodes are renumbered from 0 in coarse-node order, then in unit-node order within each
    topology, and every node inside 'bonds' is rewritten to match
    """
    new_id: dict[Address, int] = {}
    G = Graph()

    # every component's units and internal edges
    for coarse_node, topology in coarse.nodes(data=NODE_TOPOLOGY):
        if topology is None:
            raise TopologyValidationError(f"Node {coarse_node!r} isn't resolved; resolve the component graph first")
        for unit_node, key in topology.nodes(data=NODE_UNIT):
            new_id[(coarse_node, unit_node)] = len(new_id)
            G.add_node(new_id[(coarse_node, unit_node)], **{NODE_UNIT: key})

        for a, b, bonds in topology.edges(data=EDGE_BONDS):
            local = {node: new_id[(coarse_node, node)] for node in (a, b)}
            G.add_edge(local[a], local[b], **{EDGE_BONDS: tuple(
                ((local[na], pa), (local[nb], pb)) for (na, pa), (nb, pb) in bonds
            )})

    # the joins between components
    for edge_u, edge_v, bonds in coarse.edges(data=EDGE_BONDS):
        if not bonds:
            raise TopologyValidationError(f"Join ({edge_u!r}, {edge_v!r}) has no bonds; resolve the component graph first")

        rewritten = tuple(((new_id[addr_a], pa), (new_id[addr_b], pb)) for (addr_a, pa), (addr_b, pb) in bonds)
        a, b = rewritten[0][0][0], rewritten[0][1][0]
        if G.has_edge(a, b):
            raise TopologyValidationError(
                f"Join ({edge_u!r}, {edge_v!r}) lands on units {a} and {b}, which already share an edge"
            )
        G.add_edge(a, b, **{EDGE_BONDS: rewritten})

    return G


# Checks
def validate_component_graph(coarse: Graph, components: Mapping[ComponentLabel, object]) -> None:
    """Check a component graph's schema: known components, well-formed joins (resolved or not)"""
    for node, label in coarse.nodes(data=NODE_COMPONENT):
        if label not in components:
            raise TopologyValidationError(f"Node {node!r} names unknown component {label!r}")
    for u, v, joins in coarse.edges(data=EDGE_JOINS):
        _unpack_joins(joins, (u, v))
