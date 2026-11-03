"""Unit tests for multiscale-aware neighbor connections between on MuPT Primitives"""

import pytest

from mupt.mupr.primitives.primitive import Primitive
from mupt.mupr.primitives.root import RootPrimitive
from mupt.mupr.primitives.composite import CompositePrimitive
from mupt.mupr.primitives.simple.simple import SimplePrimitive
from mupt.mupr.primitives.simple.atomic import AtomicPrimitive
from mupt.mupr.connection.connectors import Connector
from mupt.chemistry.core import ELEMENTS


# Data security
@pytest.mark.parametrize(
    "prim",
    [
        RootPrimitive(),
        CompositePrimitive(),
        SimplePrimitive(),
        AtomicPrimitive(ELEMENTS[2]),
    ],
)
def test_frozen_add_connectors(prim: Primitive) -> None:
    """Test that connector addition is blocked by freezing it on any Primitive"""
    prim.freeze_connections()
    with pytest.raises(AttributeError):
        prim._add_connector(Connector())


@pytest.mark.parametrize(
    "prim",
    [
        RootPrimitive(),
        CompositePrimitive(),
        SimplePrimitive(),
        AtomicPrimitive(ELEMENTS[2]),
    ],
)
def test_frozen_remove_connectors(prim: Primitive) -> None:
    """Test that connector removal is blocked by freezing it on any Primitive"""
    prim.unfreeze_connections()
    conn_addr = prim._add_connector(Connector())
    prim.freeze_connections()
    with pytest.raises(AttributeError):
        prim._remove_connector(conn_addr)


def test_frozen_connectors_propagates() -> None:
    """Test that connector modification state changes bubble up through hierarchy"""
    root = RootPrimitive()
    comp = CompositePrimitive()
    simple = SimplePrimitive()

    comp.parent = root
    simple.parent = comp

    root.freeze_connections()
    with pytest.raises(AttributeError):
        simple._add_connector(Connector())


# Connector modification
def test_simple_remove_connector_nonexistent() -> None:
    """
    Test that attempting to remove a Connector which was
    never there to begin with is caught and raises Exception
    """
    simple = SimplePrimitive()
    connector = Connector()

    with pytest.raises(KeyError):
        simple.remove_connector(connector)


# Neighbor relations
def test_potential_neighbors():
    """
    Test that all non-internal neighbors of a
    Primitive are correctly identified in a hierarchy
    """
    ...


def test_neighbors():
    """Test resolution-specific (i.e. predicate-based) neighbor selection"""
    ...
