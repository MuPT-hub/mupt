"""Unit tests for intrinsic properties of MuPT Primitives and their subtypes"""

import pytest

from numpy import eye as identity
from numpy.testing import assert_allclose

from mupt.chemistry.core import ELEMENTS
from mupt.mupr.primitives.primitive import Primitive
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.simple.atomic import AtomicPrimitive


# Primitive (base)
@pytest.mark.parametrize(
    "prim,expected_simple,expected_supports_children,expected_supports_parents",
    [
        (RootPrimitive(), False, True, False),
        (CompositePrimitive(), False, True, True),
        (SimplePrimitive(), True, False, True),
        (AtomicPrimitive(ELEMENTS[1]), True, False, True),
    ],
)
def test_hierarchy_declarations(
    prim: Primitive,
    expected_simple: bool,
    expected_supports_children: bool,
    expected_supports_parents: bool,
) -> None:
    """
    Test whether instances of particular types of Primitive correctly
    declare their capacity in a representation hierarchy
    """
    assert (
        (prim.is_simple == expected_simple)
        and (prim.supports_children == expected_supports_children)
        and (prim.supports_parents == expected_supports_parents)
    )


# Root
def test_root_default_box_vectors() -> None:
    """Test that root can store box vector array and that it has a sensible default"""
    root = RootPrimitive()

    assert_allclose(root.box_vectors, identity(3, dtype=float))


# Composite
...

# Simple
...

# Atomic
...
