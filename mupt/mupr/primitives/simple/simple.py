"""
Base concrete SimplePrimitive - any other Primitive subtypes
which behave as a simple part of a system must inherit from here
"""

from typing import ClassVar, Iterable, Optional, Self

from copy import deepcopy

from ..primitive import PrimitiveLabel, Metadata, SupportsParents
from ...connection.types import ConnectorLabel, ConnectorAddress
from ...connection.connectors import Connector
from ...connection.management import ConnectorManager, ConnectorManagerMutable
from ....geometry.shapes.shapes import BoundedTransformableShape


class SimplePrimitive(SupportsParents):
    """
    A Primitive with no internal structure
    i.e. no children, topology, or internal connections)

    Used to explicitly demarcate "leaf" Primitives in a representation hierarchy
    Also directly manages Connector which are used by Primitives higher in the hierarchy
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "SIMPLE"

    IS_SIMPLE: ClassVar[bool] = True
    SUPPORTS_CHILDREN: ClassVar[bool] = False
    SUPPORTS_PARENTS: ClassVar[bool] = True

    def __init__(
        self,
        connections: Optional[ConnectorManager | Iterable[Connector]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        label: Optional[PrimitiveLabel] = None,
        mass: Optional[float] = None,
        metadata: Optional[Metadata] = None,
    ) -> None:
        self._init(shape=shape, label=label, mass=mass, metadata=metadata)

        # TB: have to be careful in this typecheck if ConnectorManager is also Iterable
        if isinstance(connections, Iterable):
            connections = ConnectorManagerMutable(*connections)
        elif connections is None:
            connections = ConnectorManagerMutable()

        self.connections = connections
        for connector in connections.connectors:
            connector.holder = self

    def _copy_instance(self) -> Self:
        """
        Make a copy of this SimplePrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            connections=[],  # will be plumbed up in _copy_untransformed()
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone

    # Exposing Connectors
    def add_connector(
        self,
        connector: Connector,
        label: Optional[ConnectorLabel] = None,
    ) -> ConnectorAddress:
        """
        Add a new Connector to those managed by this Simple

        Automatically assigns this Simple as its holder and propagates the
        Connector up through the hierarchy if, this Simple has a parent
        """
        connector_address = self._add_connector(connector, label=label)
        self.inject_connector_into_hierarchy(connector)
        connector.holder = self  # do last, in case above fails

        return connector_address

    def remove_connector(
        self,
        connector_address: ConnectorAddress | Connector,
    ) -> Connector:
        """
        AND from being managed by this Simple itself
        AND remove a Connector from all levels of the hierarchy above this Simple
        I.e. the passed Connector will be completely removed from the managing hierarchy

        Returns the removed Connector
        """
        connector = self._remove_connector(connector_address)
        self.withdraw_connector_from_hierarchy(
            connector_address, preserve_neighbor=False
        )
        del connector.holder  # will be self, since this Simple is at end of Path

        return connector
