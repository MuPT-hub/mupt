'''Utilities for interfacing with the anytree library (https://anytree.readthedocs.io/en/latest/)'''

from typing import Type, Union

from anytree import NodeMixin
from anytree.render import (
    RenderTree,
    AbstractStyle,
    AsciiStyle,
    ContStyle,
    ContRoundStyle,
    DoubleStyle,
)
from networkx import DiGraph

# type unions to placate linter (AbstractStyle/Type[AbstractStyle]
# flags subclasses due to non-empty __init__ defaults in their impls)
ConcreteStyle = Union[*AbstractStyle.__subclasses__()]


# Rendering and printing trees
## DEV: add any other common aliases here, as all-lowercase
RENDER_STYLE_ALIASES : dict[Type[ConcreteStyle], tuple[str, ...]] = {  
    AsciiStyle : (
        'asc',
        'ascii',
        'asciistyle',
        'ascii_style',
    ),
    ContStyle : (
        'cont',
        'contstyle',
        'cont_style',
    ),
    ContRoundStyle : (
        'round',
        'countround',
        'controundstyle',
        'cont_round_style',
    ),
    DoubleStyle : (
        'dub',
        'double',
        'doublestyle',
        'double_style',
    ),    
}
RENDER_STYLES_BY_ALIAS : dict[str, ConcreteStyle] = { 
    alias : stypetype()
        for stypetype, aliases in RENDER_STYLE_ALIASES.items()
            for alias in aliases                                                   
}

def tree_render_style(style : Union[str, ConcreteStyle, Type[ConcreteStyle]]) -> ConcreteStyle:
    '''
    Obtain a render style object which can be passed on to anytree renderers
    (https://anytree.readthedocs.io/en/latest/api/anytree.render.html)
    '''
    if isinstance(style, ConcreteStyle):
        return style
    elif isinstance(style, str):
        try:
            return RENDER_STYLES_BY_ALIAS[style.lower()]
        except KeyError:
            raise ValueError(f'Unrecognized tree render style string: "{style}"')
    elif issubclass(style, ConcreteStyle):
        return style()
    else:
        raise TypeError(f'Unsupported type for tree render style: {type(style)}')
    
# Conversion
def tree_to_networkx(node : NodeMixin) -> DiGraph:
    '''Convert a tree into a directed NetworkX graph'''
    raise NotImplementedError