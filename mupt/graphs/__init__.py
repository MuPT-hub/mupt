"""
Utilities for generating, inspecting, and visualizing graphs

Define adjacency topologies for connected systems, like covalently-bonded molecules
"""

from .graphtypes import (
    Graph as Graph,
    DiGraph as DiGraph,
    MultiGraph as MultiGraph,
    Node as Node,
    Edge as Edge,
    MultiEdge as MultiEdge,
    GraphEdge as GraphEdge,
    GraphPositions as GraphPositions,
    GraphLayout as GraphLayout,
)
from .properties import (
    is_indiscrete as is_indiscrete,
    is_trivial as is_trivial,
    is_empty as is_empty,
    is_unbranched as is_unbranched,
    is_branched as is_branched,
    is_linear as is_linear,
    termini as termini,
    leaves as leaves,
)
from .generators import (
    path_graph as path_graph,
    noodle_graph as noodle_graph,
    balanced_dendrimer_graph as balanced_dendrimer_graph,
)
