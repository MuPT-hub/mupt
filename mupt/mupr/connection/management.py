"""Managed collection of Connectors - used to outsource business logic from Primitive"""

from typing import (
    Collection,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
)
from types import MappingProxyType

from .connectors import Connector
from .types import (
    ConnectorAddress,
    ConnectorLabel,
    ConnectorLabeller,
)


def connector_address_flexible(conn: ConnectorAddress | Connector) -> ConnectorAddress:
    """
    Cast method which allows methods expecting ConnectorAddresses
    to also accept the Connector instances themselves
    """
    if isinstance(conn, Connector):
        return conn.address
    elif isinstance(conn, Hashable):
        return conn
    else:
        raise TypeError(
            f"Cannot interpret object of type '{type(conn).__name__}' "
            "as address of a Connector"
        )


class ConnectorManager(Protocol):
    """Interface for generic connector managment object"""

    connectors: Collection[Connector]
    connectors_free: Collection[Connector]
    connectors_bound: Collection[Connector]
    connectors_by_addr: Mapping[ConnectorAddress, Connector]

    def connector(self, conn_addr: ConnectorAddress) -> Connector:
        """Retrieve a particular Connector by its unique address"""
        # N.B.: not using dict.get() to make KeyErrors explicit
        return self.connectors_by_addr[conn_addr]

    def add_connector(
        self,
        conn: Connector,
        label: Optional[ConnectorLabel | ConnectorLabeller] = None,
    ) -> None:
        """Designate a Connector to be managed here"""
        ...

    def remove_connector(
        self,
        conn_addr: ConnectorAddress | Connector,
    ) -> Connector:
        """Declare a Connector to be no longer managed here"""
        ...

    # default implementations, for when explicitly inherited
    @property
    def functionality(self) -> int:
        """
        Maximum number of additional connections the
        collection of Connectors managed here could make
        """
        return len(self.connectors_free)

    # DEV: well-defined even for non-atomic systems,
    # Primitives since Connectors store BondType info
    @property
    def valence(self) -> int:
        """
        Electronic valence of the Primitive, i.e. the total bond order
        of all external-facing Connectors on this Primitive
        """
        total_bond_order: float = sum(conn.bond_order for conn in self.connectors)
        return round(total_bond_order)

    chemical_valence = electronic_valence = valence  # aliases for convenience


class HoldsConnectors(Protocol):
    """
    Type indicator for another class which is in some sense a 'proprietor' of
    a collection of Connectors, but employs a ConnectorManager to manage them
    """

    connections: ConnectorManager


# Concrete ConnectorManager types
class ConnectorManagerFrozen(ConnectorManager):
    """ConnectorManager which does not permit mutation to connectivity after creation"""

    _connectors_all: tuple[Connector, ...]
    _connectors_free: tuple[Connector, ...]
    _connectors_bound: tuple[Connector, ...]
    _connectors_by_addr: MappingProxyType[ConnectorAddress, Connector]

    def __new__(
        cls,
        *connectors: Connector,
        # TODO: provide optimization short-circuit to allow
        # making use of known free/bound designations
        connectors_free: Optional[Iterable[Connector]] = None,
        connectors_bound: Optional[Iterable[Connector]] = None,
    ) -> "ConnectorManagerFrozen":
        """Pre-compute connector properties prior to ConnectorManager creation"""
        obj = super(ConnectorManagerFrozen, cls).__new__(cls)
        obj._connectors_all = tuple(connectors)
        obj._connectors_by_addr = MappingProxyType({
            conn.address: conn for conn in connectors
        })

        connectors_free_accum: list[Connector] = []
        connectors_bound_accum: list[Connector] = []
        for conn in connectors:
            # TB DEV: lock here is not secure as yet,
            # since one could manually unlock after init
            conn.lock()  # ensure not mutations allowed subsequently
            if conn.has_neighbor:
                connectors_bound_accum.append(conn)
            else:
                connectors_free_accum.append(conn)
        obj._connectors_free = tuple(connectors_free_accum)
        obj._connectors_bound = tuple(connectors_bound_accum)

        return obj

    @property
    def connectors_by_addr(self) -> Mapping[ConnectorAddress, Connector]:
        """
        Mapping from the addresses of Connectors managed
        here to the Connector instances themselves
        """
        return self._connectors_by_addr

    @property
    def connectors(self) -> tuple[Connector, ...]:
        """All Connectors (either free or bound) managed here"""
        return self._connectors_all

    @property
    def connectors_free(self) -> tuple[Connector, ...]:
        """Connectors whose have not yet been assigned a neighbor"""
        return self._connectors_free

    @property
    def connectors_bound(self) -> tuple[Connector, ...]:
        """
        Connectors (originating from children as they must) which are
        bound and whose neighbor is also a child of this Composite
        """
        return self._connectors_bound

    def add_connector(  # noqa: D102
        self,
        conn: Connector,
        label: Optional[ConnectorLabel | ConnectorLabeller] = None,
    ) -> None:
        # TB: docstring inherited from ConnectorManager base
        raise AttributeError(
            f"Cannot add Connector to immutable {type(self).__name__} object"
        )

    def remove_connector(  # noqa: D102
        self,
        conn_addr: ConnectorAddress | Connector,
    ) -> Connector:
        # TB: docstring inherited from ConnectorManager base
        raise AttributeError(
            f"Cannot remove Connector from immutable {type(self).__name__} object"
        )


class ConnectorManagerMutable(ConnectorManager):
    """
    ConnectorManager with mutable connections
    Necessary for configuring initial connectivity
    """

    def __init__(
        self,
        *connectors: Connector,
        default_label: Hashable = "CONN",
    ) -> None:
        self.connectors_by_addr: dict[ConnectorAddress, Connector] = {}
        for conn in connectors:
            conn.unlock()
            self.add_connector(conn)

    def add_connector(  # noqa: D102
        self,
        conn: Connector,
        label: Optional[ConnectorLabel | ConnectorLabeller] = None,
    ) -> None:
        # TB: docstring inherited from ConnectorManager base

        self.connectors_by_addr[conn.addr] = conn
        # TODO: label to be used for UniqueRegistry
        # registration to give human-readable handle

    def remove_connector(  # noqa: D102
        self,
        conn_addr: ConnectorAddress | Connector,
    ) -> Connector:
        """Declare a Connector to be no longer managed here"""
        return self.connectors_by_addr.pop(connector_address_flexible(conn_addr))

    @property
    def connectors(self) -> tuple[Connector, ...]:
        """All Connectors (either free or bound) managed here"""
        return tuple(self.connectors_by_addr.values())

    # DEV: opting for linear search each time (rather than dynamically-updating list)
    # since Connectors might change neighbors during bond linking (checks when called)
    @property
    def connectors_free(self) -> tuple[Connector, ...]:
        """Managed Connectors which have no assigned neighbor"""
        return tuple(conn for conn in self.connectors if not conn.has_neighbor)

    @property
    def connectors_bound(self) -> tuple[Connector, ...]:
        """Managed Connectors which have no assigned neighbor"""
        return tuple(conn for conn in self.connectors if conn.has_neighbor)
