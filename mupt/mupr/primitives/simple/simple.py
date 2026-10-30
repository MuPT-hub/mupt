"""
Base concrete SimplePrimitive - any other Primitive subtypes
which behave as a simple part of a system must inherit from here
"""

from typing import ClassVar, Iterable, Optional, Self

from copy import deepcopy

from ..primitive import PrimitiveLabel, SupportsParents
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
        metadata: Optional[Metadata] = None,
    ) -> None:
        self._init(shape=shape, label=label, metadata=metadata)

        # TB: have to be careful in this typecheck if ConnectorManager is also Iterable
        if isinstance(connections, Iterable):
            connections = ConnectorManagerMutable(*connections)
        elif connections is None:
            connections = ConnectorManagerMutable()

        self.connections = connections
        for connector in connections.connectors:
            connector.holder = self

    # Copying
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
