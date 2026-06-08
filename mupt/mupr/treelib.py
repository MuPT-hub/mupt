'''Utilities for interfacing with the anytree library (https://anytree.readthedocs.io/en/latest/)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Union, Type

from anytree import Node, NodeMixin
from anytree.render import (
    RenderTree,
    AbstractStyle,
    AsciiStyle,
    ContStyle,
    ContRoundStyle,
    DoubleStyle,
)
from networkx import DiGraph


# Rendering and printing trees
_render_style_aliases : dict[AbstractStyle, tuple[str]] = {  # add any other common aliases here, as all-lowercase
    AsciiStyle : ('asc', 'ascii', 'asciistyle', 'ascii_style'),
    ContStyle : ('cont', 'contstyle', 'cont_style'),
    ContRoundStyle : ('round', 'countround', 'controundstyle', 'cont_round_style'),
    DoubleStyle : ('dub', 'double', 'doublestyle', 'double_style'),    
}
RENDER_STYLE_ALIASES : dict[str, AbstractStyle] = { 
    alias : stypetype()
        for stypetype, aliases in _render_style_aliases.items()
            for alias in aliases                                                   
}

def flexible_treerender_style(style : Union[str, AbstractStyle, Type[AbstractStyle]]) -> AbstractStyle: # TODO: write tests
    '''
    Obtain a render style object which can be passed on to anytree renderers
    (https://anytree.readthedocs.io/en/latest/api/anytree.render.html)
    '''
    if isinstance(style, AbstractStyle):
        return style
    elif issubclass(style, AbstractStyle):
        return style()
    elif isinstance(style, str):
        style = RENDER_STYLE_ALIASES.get(style.lower(), None)
        if style is None:
            raise ValueError(f'Unrecognized tree render style string: "{style}"')
        return style
    else:
        raise TypeError(f'Unsupported type for tree render style: {type(style)}')
    
# Conversion
def tree_to_networkx(node : NodeMixin) -> DiGraph:
    '''Convert a tree into a directed NetworkX graph'''
    raise NotImplementedError