"""
Utilities for traversing and displaying trees, as well as
converting trees between different formats

Primary interactions are with anytree (https://anytree.readthedocs.io/en/latest/)
and networkx.DiGraph (https://networkx.org/documentation/stable/reference/classes/digraph.html)
"""

from .render import RENDER_STYLE_ALIASES as RENDER_STYLE_ALIASES, RENDER_STYLES_BY_ALIAS as RENDER_STYLES_BY_ALIAS, tree_render_style as tree_render_style
from .digraph import anytree_to_networkx as anytree_to_networkx, networkx_to_anytree as networkx_to_anytree
