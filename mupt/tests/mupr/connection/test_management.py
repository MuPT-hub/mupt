"""
Unit tests for connection.management, both for the abstract ConnectorManager
and its concrete ConnectorManagerFrozen and ConnectorManagerMutable implementations
"""

# ruff: noqa: D103 (undocumented public function)
from types import MappingProxyType

import pytest

# NOTE: adjust these import paths to match your package layout
from mupt.chemistry.core import BondType
from mupt.mupr.connection.connectors import Connector
from mupt.mupr.connection.management import (
    ConnectorManagerFrozen,
    ConnectorManagerMutable,
    connector_address_flexible,
)

MANAGERS = (ConnectorManagerFrozen, ConnectorManagerMutable)


def make_connectors(n_free: int = 0, n_pairs: int = 0) -> tuple[list, list]:
    """Build n_free unbound Connectors and n_pairs mutually-bound Connector pairs"""
    free = [Connector() for _ in range(n_free)]
    bound = []
    for _ in range(n_pairs):
        a, b = Connector(), Connector()
        a.neighbor = b  # default (all-zero) positions are trivially antialigned
        bound += [a, b]
    return free, bound


# connector_address_flexible
def test_address_flexible_from_connector():
    c = Connector()
    assert connector_address_flexible(c) == c.address


@pytest.mark.parametrize("addr", ["label", 0, (1, 2), None, frozenset({3})])
def test_address_flexible_hashable_passthrough(addr):
    assert connector_address_flexible(addr) is addr


@pytest.mark.parametrize("addr", [[], {}, set()])
def test_address_flexible_unhashable_raises(addr):
    with pytest.raises(TypeError, match="as address of a Connector"):
        connector_address_flexible(addr)


# Shared behavior across concrete managers
@pytest.mark.parametrize("n_free, n_pairs", [(0, 0), (2, 0), (0, 1), (2, 2)])
@pytest.mark.parametrize("manager_cls", MANAGERS)
def test_partition_and_lookup(manager_cls, n_free, n_pairs):
    free, bound = make_connectors(n_free, n_pairs)
    mgr = manager_cls(*free, *bound)

    assert mgr.connectors == tuple(free + bound)
    assert mgr.connectors_free == tuple(free)
    assert mgr.connectors_bound == tuple(bound)
    assert mgr.functionality == n_free
    assert list(mgr.connectors_by_addr) == [c.address for c in free + bound]
    assert all(mgr.connector(c.address) is c for c in free + bound)
    with pytest.raises(KeyError):
        mgr.connector(Connector().address)


@pytest.mark.parametrize(
    "bondtypes, expected",
    [
        ([], 0),
        ([BondType.UNSPECIFIED], 0),
        ([BondType.SINGLE, BondType.DOUBLE], 3),
        ([BondType.AROMATIC] * 2, 3),
        ([BondType.DOUBLE, BondType.AROMATIC], 4),  # round(3.5)
    ],
)
@pytest.mark.parametrize("manager_cls", MANAGERS)
def test_valence(manager_cls, bondtypes, expected):
    mgr = manager_cls(*(Connector(bondtype=bt) for bt in bondtypes))
    assert mgr.valence == mgr.chemical_valence == mgr.electronic_valence == expected


# ConnectorManagerFrozen specifics
def test_frozen_locks_and_mutable_unlocks():
    conns = sum(make_connectors(2, 1), [])
    frozen = ConnectorManagerFrozen(*conns)
    assert isinstance(frozen.connectors_by_addr, MappingProxyType)
    assert all(c.is_locked for c in conns)

    ConnectorManagerMutable(*conns)
    assert not any(c.is_locked for c in conns)


@pytest.mark.parametrize("method", ["add_connector", "remove_connector"])
def test_frozen_rejects_mutation(method):
    with pytest.raises(AttributeError, match="immutable ConnectorManagerFrozen"):
        getattr(ConnectorManagerFrozen(), method)(Connector())


# ConnectorManagerMutable specifics
@pytest.mark.parametrize(
    "key", [lambda c: c, lambda c: c.address], ids=["instance", "address"]
)
@pytest.mark.parametrize("via_init", [True, False], ids=["init", "add"])
@pytest.mark.parametrize(
    "label, expected",
    [
        (None, Connector.DEFAULT_LABEL),
        ("X", "X"),
        (lambda c: c.DEFAULT_LABEL * 2, "ConnConn"),
    ],
    ids=["none", "static", "callable"],
)
def test_mutable_add_remove(label, expected, via_init, key):
    c = Connector()
    if via_init:
        mgr = ConnectorManagerMutable(c, default_label=label)
    else:
        mgr = ConnectorManagerMutable()
        mgr.add_connector(c, label=label)

    assert c.label == expected
    assert mgr.connectors == (c,) and mgr.functionality == 1
    assert mgr.remove_connector(key(c)) is c
    assert mgr.connectors == () and mgr.functionality == 0
    with pytest.raises(KeyError):
        mgr.remove_connector(c)


def test_mutable_partition_tracks_rebinding():
    (a, b), _ = make_connectors(2)
    mgr = ConnectorManagerMutable(a, b)
    assert mgr.connectors_free == (a, b) and mgr.connectors_bound == ()

    a.neighbor = b
    assert mgr.connectors_free == () and mgr.connectors_bound == (a, b)

    del a.neighbor
    assert mgr.connectors_free == (a, b) and mgr.functionality == 2
