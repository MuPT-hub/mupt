"""
Realization: turning descriptive graphs into MuPT Primitives

A descriptive graph needs only two things (see mupt.polymers._common):
    node 'unit'  : the lexicon key of the repeat-unit template that node represents
    edge 'bonds' : ((node, port), (node, port)) pairs, one per bond between the edge's units

Realization copies one template per node and bonds the named ports to each other. It
deduces nothing, so it works for any graph -- linear, ladder, branched, or hand-built --
and is independent of the generator (if any) which produced the graph.
"""

from typing import Hashable, Iterable, Mapping

from networkx import Graph

from ...mupr.connection import Connector
from ...mupr.primitives import CompositePrimitive, Primitive, RootPrimitive
from ...mupr.properties import has_strict_SAAMR_depth
from ...roles import PrimitiveRole, assign_SAAMR_roles

from .._common import EDGE_BONDS, NODE_UNIT, Node, PortLabel, UnitKey
from ..exceptions import TopologyValidationError
from ..rules.faces import free_ports_by_label


# Public API
def primitive_from_descriptive_graph(
    G: Graph,
    lexicon: Mapping[UnitKey, Primitive],
    *,
    label: Hashable = "chain",
    unit_attr: str = NODE_UNIT,
    bonds_attr: str = EDGE_BONDS,
) -> CompositePrimitive:
    """
    Build one Primitive from a descriptive graph, with one child per node

    The whole graph is validated against the lexicon before anything is copied or
    bonded, so an invalid graph raises TopologyValidationError and builds nothing.
    Children are attached in the graph's node order; ports not named by any bond
    are left free (e.g. uncapped chain ends).
    """
    # check everything up front, resolving each bond to the template ports it names
    validate_descriptive_graph(G, lexicon, unit_attr=unit_attr, bonds_attr=bonds_attr)

    # copy one template per node; residue metadata comes along with each copy
    chain = CompositePrimitive(label=label)
    units: dict[Node, Primitive] = {}
    for node, key in G.nodes(data=unit_attr):
        units[node] = lexicon[key].copy()
        chain.attach_child(units[node])

    # look up every node's ports BEFORE bonding: a bonded Connector is no longer free
    ports = {node: free_ports_by_label(unit) for node, unit in units.items()}

    # validation guarantees these ports exist, are unused, and have matching bond types;
    # alignment isn't checked, since placing units relative to one another comes later
    for (node_a, port_a), (node_b, port_b) in _iter_bonds(G, bonds_attr):
        ports[node_a][port_a].bond_to(ports[node_b][port_b], check_alignment=False)

    return chain


def universe_from_descriptive_graphs(
    graphs: Iterable[Graph],
    lexicon: Mapping[UnitKey, Primitive],
    label: Hashable = "universe",
) -> RootPrimitive:
    """Build a universe with one chain per descriptive graph, labelled chain_0, chain_1, ..."""
    universe = RootPrimitive(label=label)
    for i, G in enumerate(graphs):
        universe.attach_child(primitive_from_descriptive_graph(G, lexicon, label=f"chain_{i}"))

    _assign_roles(universe)
    return universe


def combine_universes(*universes: RootPrimitive, label: Hashable = "universe") -> RootPrimitive:
    """
    Combine universes into a NEW universe holding copies of all their chains, in order

    The inputs are left unchanged (so memory use briefly doubles for large systems).
    Chains are relabelled chain_0, chain_1, ... since every universe starts at chain_0.
    """
    combined = RootPrimitive(label=label)
    chains = (chain for universe in universes for chain in universe.children)
    for i, chain in enumerate(chains):
        chain_copy = chain.copy()  # bonds between a chain's units are copied along with it
        chain_copy.label = f"chain_{i}"
        combined.attach_child(chain_copy)

    _assign_roles(combined)
    return combined


# Validation
def _iter_bonds(G: Graph, bonds_attr: str):
    """Yield every bond record in the graph, edge by edge"""
    for _, _, bonds in G.edges(data=bonds_attr):
        yield from bonds


def validate_descriptive_graph(
    G: Graph,
    lexicon: Mapping[UnitKey, Primitive],
    *,
    unit_attr: str = NODE_UNIT,
    bonds_attr: str = EDGE_BONDS,
) -> None:
    """
    Raise TopologyValidationError unless every node and bond of G can be realized

    Checks a graph against a lexicon without building anything; realization runs
    the same check first
    """
    # every node must name a template in the lexicon
    keys: dict[Node, UnitKey] = {}
    for node, data in G.nodes(data=True):
        if unit_attr not in data:
            raise TopologyValidationError(f"Node {node!r} has no {unit_attr!r} attribute")
        try:
            known = data[unit_attr] in lexicon
        except TypeError:  # unhashable key
            known = False
        if not known:
            raise TopologyValidationError(f"Node {node!r} names unit {data[unit_attr]!r}, which isn't in the lexicon")
        keys[node] = data[unit_attr]

    # port labels survive copying, so the templates' ports stand in for the copies' ports
    template_ports: dict[UnitKey, dict[PortLabel, Connector]] = {
        key: free_ports_by_label(lexicon[key]) for key in set(keys.values())
    }

    used: dict[tuple[Node, PortLabel], tuple[Node, Node]] = {}  # (node, port) -> edge using it
    for u, v, data in G.edges(data=True):
        edge = (u, v)
        if u == v:
            raise TopologyValidationError(f"Edge {edge!r} is a self-loop; a unit can't bond to itself")
        if not data.get(bonds_attr):
            raise TopologyValidationError(f"Edge {edge!r} has no {bonds_attr!r} (every edge needs at least one bond)")

        for bond in data[bonds_attr]:
            (node_a, port_a), (node_b, port_b) = _unpack_bond(bond, edge)

            # a bond on an edge must join exactly that edge's two units
            if {node_a, node_b} != {u, v}:
                raise TopologyValidationError(
                    f"Bond {bond!r} is stored on edge {edge!r} but joins nodes {node_a!r} and {node_b!r}"
                )

            for node, port in ((node_a, port_a), (node_b, port_b)):
                if port not in template_ports[keys[node]]:
                    raise TopologyValidationError(
                        f"Bond {bond!r} names port {port!r} on node {node!r}, but unit "
                        f"{keys[node]!r} has free ports {sorted(map(str, template_ports[keys[node]]))}"
                    )
                if (node, port) in used:
                    raise TopologyValidationError(
                        f"Port {port!r} on node {node!r} is bonded on edge {used[(node, port)]!r} "
                        f"and again on edge {edge!r}"
                    )
                used[(node, port)] = edge

            # both halves of a bond must agree on its type (single, double, aromatic, ...)
            type_a = template_ports[keys[node_a]][port_a].bondtype
            type_b = template_ports[keys[node_b]][port_b].bondtype
            if type_a != type_b:
                raise TopologyValidationError(f"Bond {bond!r} joins ports with different bond types ({type_a}, {type_b})")


def _unpack_bond(bond, edge) -> tuple[tuple[Node, PortLabel], tuple[Node, PortLabel]]:
    """Check that a bond record has the shape ((node, port), (node, port))"""
    try:
        (node_a, port_a), (node_b, port_b) = bond
    except (TypeError, ValueError):
        raise TopologyValidationError(
            f"Bond {bond!r} on edge {edge!r} isn't of the form ((node, port), (node, port))"
        ) from None
    return (node_a, port_a), (node_b, port_b)


# Roles
def _assign_roles(universe: RootPrimitive) -> None:
    """Assign SAAMR roles: universe -> chains (segments) -> units (residues) -> leaves (particles)"""
    # the common case: all-atom units give a strict root -> chain -> unit -> atom hierarchy
    if has_strict_SAAMR_depth(universe):
        assign_SAAMR_roles(universe)
        return

    # otherwise (e.g. units with deeper hierarchies), assign by level explicitly
    universe.role = PrimitiveRole.UNIVERSE
    for chain in universe.children:
        chain.role = PrimitiveRole.SEGMENT
        for unit in chain.children:
            unit.role = PrimitiveRole.RESIDUE
            for leaf in unit.leaves:
                leaf.role = PrimitiveRole.PARTICLE
