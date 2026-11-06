"""Unit tests for Primitive roles"""

# ruff: noqa: F401

import pytest

from itertools import product as cartesian

from mupt.mupr.primitives.roles import PrimitiveRole
from mupt.mupr.primitives.primitive import Primitive
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.simple.atomic import AtomicPrimitive, ELEMENTS


def str_to_role_pairs() -> dict[str, PrimitiveRole]:
    """Examples of various strings we expect to be associable with particular Roles"""
    return {
        "unassigned": PrimitiveRole.UNASSIGNED,
        "UNASSIGNED": PrimitiveRole.UNASSIGNED,
        "universe": PrimitiveRole.UNIVERSE,
        "UNIVERSE": PrimitiveRole.UNIVERSE,
        "segment": PrimitiveRole.SEGMENT,
        "SEGMENT": PrimitiveRole.SEGMENT,
        "residue": PrimitiveRole.RESIDUE,
        "RESIDUE": PrimitiveRole.RESIDUE,
        "particle": PrimitiveRole.PARTICLE,
        "PARTICLE": PrimitiveRole.PARTICLE,
    }


@pytest.mark.parametrize(
    "str_input,role_expected",
    [
        *str_to_role_pairs().items(),
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


@pytest.mark.parametrize(
    "prim,role_str,role_expected",
    [
        (prim, role_str, role_expected)
        for prim, (role_str, role_expected) in cartesian(
            [
                RootPrimitive(),
                CompositePrimitive(),
                SimplePrimitive(),
                AtomicPrimitive(ELEMENTS[6]),
            ],
            str_to_role_pairs().items(),
        )
    ],
)
def test_prim_role_sanctioned(
    prim: Primitive, role_str: str, role_expected: PrimitiveRole
) -> None:
    """
    Test that Primitives can be assigned role str tags
    which are resolvable to sanctioned PrimitiveRole values
    """
    prim.role = role_str
    assert prim.role_sanctioned is role_expected
