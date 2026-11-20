"""Unit tests for pre-compiled predicates on Primitives"""

import pytest

from dataclasses import dataclass

from mupt.mupr.primitives.primitive import Primitive
from mupt.mupr.primitives import predicates


@dataclass
class PrimitivePredicateExample:
    """Encapsulation for predicate tests with expected answers"""

    primitive: Primitive
    foo_expected: bool
    ...


PRIM_EXAMPLES: list[PrimitivePredicateExample] = []  # TODO: populate


@pytest.mark.parametrize("prim_example", PRIM_EXAMPLES)
def test_predicate_foo(prim_example: PrimitivePredicateExample) -> None:
    """Test that predicate <foo> evaluates as expected on several example Primitives"""
    assert predicates.is_simple(prim_example.primitive) is prim_example.foo_expected
