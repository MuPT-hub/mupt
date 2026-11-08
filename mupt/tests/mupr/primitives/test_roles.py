"""Unit tests for Primitive roles"""

# ruff: noqa: F401

import pytest

from itertools import product as cartesian

from mupt.mupr.primitives.roles import PrimitiveRole
from mupt.mupr.primitives.primitive import Primitive, DEFAULT_ROLE
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


def primitive_instances() -> tuple[Primitive, ...]:
    """Example role-less Primitive instances"""
    return (
        RootPrimitive(),
        CompositePrimitive(),
        SimplePrimitive(),
        AtomicPrimitive(ELEMENTS[6]),
    )


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
    "prim,role_obj",
    [
        (prim, role)
        for prim, role in cartesian(
            primitive_instances(), [role for role in PrimitiveRole]
        )
    ],
)
def test_role_instances_normalized(prim: Primitive, role_obj: PrimitiveRole) -> None:
    """
    Test that Primitive.role also accepts PrimitiveRole
    instances and converts them into string form internally
    """
    # underscoring is deliberate; use setter to assign but check private value otherwise
    prim.role = role_obj
    assert prim._role == role_obj.value


@pytest.mark.parametrize(
    "prim,role_str,role_expected",
    [
        (prim, role_str, role_expected)
        for prim, (role_str, role_expected) in cartesian(
            primitive_instances(),
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
