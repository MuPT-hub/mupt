"""
Quick plots of descriptive graphs, for inspection while building systems

    draw_topology        : a unit graph (what generate_topologies() returns)
    draw_component_graph : a composite's component graph (generate_component_graph())

matplotlib is imported only when something is drawn, so the rest of mupt.polymers
doesn't depend on it.
"""

from typing import Mapping, Optional

import networkx as nx
from networkx import Graph

from .._common import EDGE_BONDS, EDGE_JOINS, NODE_COMPONENT, NODE_TOPOLOGY, NODE_UNIT, FaceSpec, FixedFace, UnitLabel


def _pyplot():
    """matplotlib.pyplot, imported on first use"""
    import matplotlib.pyplot as plt
    return plt


def _colours_by(values: list) -> list:
    """One tab10 colour per distinct value, in order of the values' text"""
    plt = _pyplot()
    kinds = sorted({str(value) for value in values})
    colour = {kind: plt.cm.tab10(i % 10) for i, kind in enumerate(kinds)}
    return [colour[str(value)] for value in values]


# Unit graphs
def draw_topology(G: Graph, ax=None, show_ports: bool = False):
    """Quick plot of a descriptive (unit) graph: nodes labelled and coloured by unit"""
    if any(NODE_UNIT not in data for _, data in G.nodes(data=True)):
        if any(NODE_COMPONENT in data for _, data in G.nodes(data=True)):
            raise TypeError(
                "This is a component graph; draw it with draw_component_graph(), "
                "or flatten it first with flatten_component_graph()"
            )
        raise TypeError(f"Every node of a unit graph needs a {NODE_UNIT!r} attribute")
    ax = ax or _pyplot().gca()

    # label = last part of the unit label, plus its tactic config if any
    def short(key):
        label, config = key
        name = label[-1] if isinstance(label, tuple) else label
        return name if config is None else f"{name}{'+' if config == 1 else '-'}"

    labels = {n: short(key) for n, key in G.nodes(data=NODE_UNIT)}

    # one colour per unit label, so blocks/components stand out
    node_colours = _colours_by([key[0] for _, key in G.nodes(data=NODE_UNIT)])

    pos = nx.kamada_kawai_layout(G)  # good for chains, stars and combs
    nx.draw(G, pos, ax=ax, labels=labels, node_color=node_colours, node_size=500, font_size=8)

    if show_ports:  # e.g. "2-1" for a tail-to-head bond, one line per bond (ladders have two)
        edge_labels = {
            (u, v): "\n".join(f"{pu}-{pv}" for (_, pu), (_, pv) in bonds)
            for u, v, bonds in G.edges(data=EDGE_BONDS)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)
    return ax


# Component graphs
def _anchor_text(anchor) -> str:
    """How an unresolved join anchor reads on a plot"""
    if anchor is None:
        return "any"
    if isinstance(anchor, FixedFace):
        return f"{anchor.face}@{anchor.node}"  # face of one node of an already resolved component
    return str(anchor)  # a site pattern


def _face_of(C: Graph, address, port, faces: Optional[Mapping[UnitLabel, FaceSpec]]):
    """Name of the face holding a bonded port, or the port itself if faces aren't known"""
    coarse_node, unit_node = address
    if faces is None:
        return port
    label = C.nodes[coarse_node][NODE_TOPOLOGY].nodes[unit_node][NODE_UNIT][0]
    for name, ports in faces.get(label, {}).items():
        if port in ports:
            return name
    return port


def _join_text(C: Graph, u, v, faces) -> str:
    """Edge label for one join: anchors if unresolved, faces (or ports) bonded if resolved"""
    data = C.edges[u, v]
    (up, anchor_up), (_, anchor_down) = data[EDGE_JOINS]
    bonds = data.get(EDGE_BONDS)
    if bonds is None:
        return f"{_anchor_text(anchor_up)} → {_anchor_text(anchor_down)}"

    # write each bond upstream first, since the edge may be stored either way round
    lines = []
    for end_a, end_b in bonds:
        if end_a[0][0] != up:
            end_a, end_b = end_b, end_a
        (addr_up, port_up), (addr_down, port_down) = end_a, end_b
        lines.append(f"{_face_of(C, addr_up, port_up, faces)} → {_face_of(C, addr_down, port_down, faces)}")

    # a multi-strand join bonds several ports of the same two faces: show each face pair once
    return "\n".join(dict.fromkeys(lines)) if faces is not None else "\n".join(lines)


def draw_component_graph(
    C: Graph,
    ax=None,
    show_joins: bool = True,
    faces: Optional[Mapping[UnitLabel, FaceSpec]] = None,
):
    """
    Quick plot of a component graph: one node per component, coloured by component label

    Nodes show their component label, and their number of units once resolved. With
    show_joins, each edge shows its join, upstream first: the anchors ("any", a site
    pattern, or face@node for a fixed face) if unresolved, or the faces actually bonded if
    resolved. Pass the composite's faces (gen.faces) to name those faces; without them the
    bonded port labels are shown instead.

    To see the units themselves, draw flatten_component_graph(C) with draw_topology().
    """
    if any(NODE_COMPONENT not in data for _, data in C.nodes(data=True)):
        raise TypeError(
            f"Every node of a component graph needs a {NODE_COMPONENT!r} attribute; "
            "for unit graphs use draw_topology()"
        )
    ax = ax or _pyplot().gca()

    labels = {}
    for node, data in C.nodes(data=True):
        topology = data.get(NODE_TOPOLOGY)
        size = "" if topology is None else f"\n({topology.number_of_nodes()} units)"
        labels[node] = f"{data[NODE_COMPONENT]}{size}"
    node_colours = _colours_by([label for _, label in C.nodes(data=NODE_COMPONENT)])

    pos = nx.kamada_kawai_layout(C) if C.number_of_nodes() > 1 else {node: (0, 0) for node in C}
    nx.draw(C, pos, ax=ax, labels=labels, node_color=node_colours, node_size=1500, font_size=8)

    if show_joins:
        edge_labels = {(u, v): _join_text(C, u, v, faces) for u, v in C.edges}
        nx.draw_networkx_edge_labels(C, pos, edge_labels=edge_labels, ax=ax, font_size=7)
    return ax
