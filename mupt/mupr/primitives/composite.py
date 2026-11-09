"""
Concrete Primitive subtype for 'composites', which represent an intermediate
resolution Primitives which can be both parent and child of other Primitives

Used as the "connective tissue" between the root and simples of a hierarachy
"""

from typing import ClassVar, Iterable, Optional, Self

from copy import deepcopy

from .primitive import PrimitiveLabel, Metadata, SupportsChildren, SupportsParents
from ..connection.management import ConnectorManagerMutable
from ...geometry.shapes.shapes import BoundedTransformableShape


class CompositePrimitive(SupportsChildren, SupportsParents):
    """
    Primitive representing intermediate levels of organization in a chemical system;
    In a representation hierarchy, always lives between Roots and Simples
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "COMPOSITE"

    IS_SIMPLE: ClassVar[bool] = False
    SUPPORTS_CHILDREN: ClassVar[bool] = True
    SUPPORTS_PARENTS: ClassVar[bool] = True

    def __init__(
        self,
        children: Optional[Iterable[SupportsParents]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        label: Optional[PrimitiveLabel] = None,
        mass: Optional[float] = None,
        metadata: Optional[Metadata] = None,
    ) -> None:
        self._init(shape=shape, label=label, mass=mass, metadata=metadata)
        self.connections = ConnectorManagerMutable()
        # N.B.: can't call _init_children before _frozen_hierarchy is set
        self._init_children(children)

    # Copying
    def _copy_instance(self) -> Self:
        """
        Make a copy of this CompositePrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            children=[],
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone
