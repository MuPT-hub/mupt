"""
Utilities for generating, inspecting, and visualising graphs which encode
incidence topologies for connected systems, like covalently-bonded molecules
"""

from .graphtypes import (
    Graph as Graph,
    DiGraph as DiGraph,
    MultiGraph as MultiGraph,
    Node as Node,
    Edge as Edge,
    MultiEdge as MultiEdge,
    GraphEdge as GraphEdge,
    GraphLike as GraphLike,
    GraphPositions as GraphPositions,
    GraphLayout as GraphLayout,
)
from .invariants import (
    is_indiscrete as is_indiscrete,
    is_trivial as is_trivial,
    is_empty as is_empty,
    is_unbranched as is_unbranched,
    is_acyclic as is_acyclic,
    is_branched as is_branched,
    is_linear as is_linear,
    termini as termini,
    leaves as leaves,
    components as components,
    chains as chains,
    canonical_form_graph as canonical_form_graph,
)
from .generators import (
    path_graphs as path_graphs,
    path_graph_from_sequence as path_graph_from_sequence,
    noodle_graph as noodle_graph,
    balanced_dendrimer_graph as balanced_dendrimer_graph,
)
# DEV: excluding .visualization utils at toplevel
# to decouple from matplotlib/pygraphviz dependencies
