"""Unit tests for tree rendering"""

import pytest
from typing import Union

from anytree.render import (
    AsciiStyle,
    ContStyle,
    ContRoundStyle,
    DoubleStyle,
)

from mupt.trees.render import ConcreteStyle, tree_render_style


@pytest.mark.parametrize(
    "style,style_type_expected",
    [
        # ASCII
        (AsciiStyle(), AsciiStyle),
        (AsciiStyle, AsciiStyle),
        ("asc", AsciiStyle),
        ("ascii", AsciiStyle),
        ("ASCII", AsciiStyle),
        ("aScIi", AsciiStyle),
        ("asciistyle", AsciiStyle),
        ("ascii_style", AsciiStyle),
        # Continued
        (ContStyle(), ContStyle),
        (ContStyle, ContStyle),
        ("cont", ContStyle),
        ("continued", ContStyle),
        ("CONTINUED", ContStyle),
        ("cOnTiNuEd", ContStyle),
        ("contstyle", ContStyle),
        ("cont_style", ContStyle),
        # Continued-round
        (ContRoundStyle(), ContRoundStyle),
        (ContRoundStyle, ContRoundStyle),
        ("round", ContRoundStyle),
        ("contround", ContRoundStyle),
        ("CONTROUND", ContRoundStyle),
        ("cOnTrOuNd", ContRoundStyle),
        ("controundstyle", ContRoundStyle),
        ("cont_round_style", ContRoundStyle),
        # Double lines
        (DoubleStyle(), DoubleStyle),
        (DoubleStyle, DoubleStyle),
        ("dub", DoubleStyle),
        ("double", DoubleStyle),
        ("DOUBLE", DoubleStyle),
        ("dOuBlE", DoubleStyle),
        ("doublestyle", DoubleStyle),
        ("double_style", DoubleStyle),
    ],
)
def test_tree_render_style(
    style: Union[str, ConcreteStyle, type[ConcreteStyle]],
    style_type_expected: type[ConcreteStyle],
) -> None:
    """
    Test that flexible tree render style fetch returns
    expected style object for all supported inputs
    """
    # DEV: regrettably, AbstractStyle subtypes don't implement __eq__, so
    # distinct instances cannot be compared; need to compare types instead
    assert type(tree_render_style(style)) is style_type_expected
