"""Managed collection of Connectors - used to outsource business logic from Primitive"""

from typing import (
    Collection,
    Mapping,
    Optional,
    Protocol,
)

from .connectors import Connector
from .types import (
    ConnectorAddress,
    ConnectorLabelLike,
)


class ConnectorManager(Protocol):
    """Interface for generic connector managment object"""

    connectors: Collection[Connector]
    connectors_free: Collection[Connector]
    connectors_bound: Collection[Connector]
    connectors_by_addr: Mapping[ConnectorAddress, Connector]

    def connector(self, connector_address: ConnectorAddress) -> Connector:
        """Retrieve a particular Connector by its unique address"""
        # N.B.: not using dict.get() to make KeyErrors explicit
        return self.connectors_by_addr[connector_address]

    def add_connector(
        self,
        connector: Connector,
        label: Optional[ConnectorLabelLike] = None,
    ) -> None:
        """Designate a Connector to be managed here"""
        ...

    def remove_connector(
        self,
        connector_address: ConnectorAddress | Connector,
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
        total_bond_order: float = sum(
            connector.bond_order for connector in self.connectors
        )
        return round(total_bond_order)

    chemical_valence = electronic_valence = valence  # aliases for convenience


class HoldsConnectors(Protocol):
    """
    Type indicator for another class which is in some sense a 'proprietor' of
    a collection of Connectors, but employs a ConnectorManager to manage them
    """

    connections: ConnectorManager
