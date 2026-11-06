"""Unit tests for Primitive roles"""

# ruff: noqa: F401

import pytest

from mupt.mupr.primitives.roles import PrimitiveRole
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.simple.atomic import AtomicPrimitive


@pytest.mark.parametrize(
    "str_input,role_expected",
    [
        ("unassigned", PrimitiveRole.UNASSIGNED),
        ("UNASSIGNED", PrimitiveRole.UNASSIGNED),
        ("universe", PrimitiveRole.UNIVERSE),
        ("UNIVERSE", PrimitiveRole.UNIVERSE),
        ("segment", PrimitiveRole.SEGMENT),
        ("SEGMENT", PrimitiveRole.SEGMENT),
        ("residue", PrimitiveRole.RESIDUE),
        ("RESIDUE", PrimitiveRole.RESIDUE),
        ("particle", PrimitiveRole.PARTICLE),
        ("PARTICLE", PrimitiveRole.PARTICLE),
        pytest.param(
            None,
            None,
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="NoneType is not a valid role key",
                strict=True,
            ),
        ),
        pytest.param(
            73,
            None,
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="int is not a valid role key",
                strict=True,
            ),
        ),
        pytest.param(
            ["universe"],
            None,
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="list is not a valid role key",
                strict=True,
            ),
        ),
    ],
)
def test_role_from_str(str_input: str, role_expected: PrimitiveRole) -> None:
    """Test that Roles can be initialized from non mupt-specific strings"""
    role: PrimitiveRole = PrimitiveRole.from_str(str_input)
    assert role is role_expected
