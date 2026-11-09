"""Unit tests for copying entire representation hierarchies"""

import pytest


from anytree.iterators import PreOrderIter

from mupt.chemistry.core import ELEMENTS
from mupt.mupr.connection.connectors import Connector, AttachmentPoint, BondType
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.simple.atomic import AtomicPrimitive


# Fixtures
@pytest.fixture(scope="function")
def hierarchy_example() -> RootPrimitive:
    """
    A moderately-complex but still-small hierarchy which contains
    all Primitive subtypes and varied parent/child relationships
    """
    # Connectors
    conn_prototype_0 = Connector(
        anchor=AttachmentPoint({1}),
        linker=AttachmentPoint({2}),
        bondtype=BondType.DOUBLE,
    )
    conn_prototype_1 = Connector(
        anchor=AttachmentPoint({1}),
        linker=AttachmentPoint({2}),
        bondtype=BondType.AROMATIC,
    )

    # Primitive parts
    ## TB: not relying on auto-linking; painstakingly connected by hand in manner
    ## I've ensured is valid to isolate just the copying methods for testing here
    root = RootPrimitive()
    comp_0 = CompositePrimitive()
    comp_1 = CompositePrimitive()
    comp_2 = CompositePrimitive()
    conn_00_0o = conn_prototype_0.copy()
    simple_0 = SimplePrimitive(connections=(conn_00_0o,))

    conn_10_0c = conn_prototype_0.counterpart()
    conn_11_1o = conn_prototype_1.copy()
    conn_12_0o = conn_prototype_0.copy()
    conn_13_0c = conn_prototype_0.counterpart()
    simple_1 = SimplePrimitive(
        connections=(conn_10_0c, conn_11_1o, conn_12_0o, conn_13_0c)
    )
    conn_20_1o = conn_prototype_1.counterpart()
    conn_21_0o = conn_prototype_0.copy()
    conn_22_oc = conn_prototype_0.counterpart()
    simple_2 = SimplePrimitive(connections=(conn_20_1o, conn_21_0o, conn_22_oc))

    conn_30_1c = conn_prototype_1.counterpart()
    conn_31_1c = conn_prototype_1.counterpart()
    atom_1 = AtomicPrimitive(element=ELEMENTS[6], connections=(conn_30_1c, conn_31_1c))

    # Assembly
    simple_0.parent = root
    comp_0.parent = root
    comp_1.parent = comp_0
    simple_1.parent = comp_1
    comp_2.parent = root
    comp_2.children = [simple_2, atom_1]  # also test that this mechanism works

    # Connection
    simple_0.connect_neighbor(
        simple_1, our_connector=conn_00_0o, their_connector=conn_10_0c
    )
    simple_1.connect_neighbor(
        simple_2, our_connector=conn_12_0o, their_connector=conn_22_oc
    )
    simple_1.connect_neighbor(
        simple_2, our_connector=conn_13_0c, their_connector=conn_21_0o
    )
    simple_1.connect_neighbor(
        atom_1, our_connector=conn_11_1o, their_connector=conn_30_1c
    )
    simple_2.connect_neighbor(
        atom_1, our_connector=conn_20_1o, their_connector=conn_31_1c
    )

    return root


# Tests proper
# TB: tests here are slightly brittle, since they assume copied
# Connectors and Primitives will be in same order as in the original.
# Revisit if that assumption becomes invalid (these tests will start failing)
def test_primitive_copy_hierarchy(hierarchy_example: RootPrimitive) -> None:
    """Test that hierarchy of copy is isomorphic to that of the original"""
    hierarchy_copy = hierarchy_example.copy()
    for prim_orig, prim_copy in zip(
        PreOrderIter(hierarchy_example),
        PreOrderIter(hierarchy_copy),
    ):
        assert (
            # TODO: implement __eq__ on all Primitive subtypes to
            # provide more robust comparison over select attribute
            type(prim_orig) is type(prim_copy)
            and prim_orig.address != prim_copy.address
        )


def test_primitive_copy_connectors(hierarchy_example: RootPrimitive) -> None:
    """Test that Connectors on copy are analogous to original WITHOUT being identical"""
    hierarchy_copy = hierarchy_example.copy()
    for conn_orig, conn_copy in zip(
        hierarchy_example.connectors,
        hierarchy_copy.connectors,
    ):
        assert (
            Connector.fungible_with(conn_copy, conn_orig)
            and (conn_copy is not conn_orig)  # equivalent but not identical
        )


@pytest.mark.parametrize(
    "role",
    [
        "unassigned",
        "segment",
        "SEGMENT",
        "non-sanctioned-valid-str",
    ],
)
def test_role_kept_on_copy(role: str) -> None:
    """Test that copied Primtiives preserve the role of their prototype"""
    prim = CompositePrimitive(label="foo", role=role)
    copy = prim.copy()

    assert copy.role == prim.role
