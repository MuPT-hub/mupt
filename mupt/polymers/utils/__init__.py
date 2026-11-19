"""
Unit and graph machinery behind the generators

    lexicon   : building repeat-unit and end-cap templates
    stereo    : stereo variants of units, for tacticity
    realize   : turning descriptive graphs into MuPT Primitives
    compose   : resolving and flattening the component graphs of composites
    visualize : quick plots of unit and component graphs (needs matplotlib)

Only the functions useful on hand-built or hand-edited graphs are re-exported here.
"""

from .realize import (
    primitive_from_descriptive_graph as primitive_from_descriptive_graph,
    universe_from_descriptive_graphs as universe_from_descriptive_graphs,
    combine_universes as combine_universes,
    validate_descriptive_graph as validate_descriptive_graph,
)
from .compose import (
    resolve_component_graph as resolve_component_graph,
    flatten_component_graph as flatten_component_graph,
    validate_component_graph as validate_component_graph,
)
from .visualize import (
    draw_topology as draw_topology,
    draw_component_graph as draw_component_graph,
)
