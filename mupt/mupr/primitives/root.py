"""
Concrete Primitive subtype for 'roots', which represent a
distinguished, coarsest-resolution Primitive in a complete hierarchy

Most commonly used to represent an entire chemical system or universe
"""

from typing import ClassVar, Iterable, Optional, Self

from copy import deepcopy

from numpy import eye as identity_matrix

from .primitive import PrimitiveLabel, Metadata, SupportsChildren, SupportsParents
from ..connection.management import ConnectorManagerMutable
from ...geometry.arraytypes import Array3x3
from ...geometry.shapes.shapes import BoundedTransformableShape


class RootPrimitive(SupportsChildren):
    """
    Base of a hierarchy tree - no Primitives can exist above (i.e. own) this one
    Used to store system-wide metadata, as well as provide hand-off point for interfaces
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "ROOT"

    IS_SIMPLE: ClassVar[bool] = False
    SUPPORTS_CHILDREN: ClassVar[bool] = True
    SUPPORTS_PARENTS: ClassVar[bool] = False

    # TODO: eventually replace this with a MuPT Box-type object, once that's settled on
    box_vectors: Array3x3

    def __init__(
        self,
        box_vectors: Optional[Array3x3] = None,
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

        # system-wide info specific to Root instances
        if box_vectors is None:
            # TODO: associate units (once a standard has been decided upon)
            box_vectors = identity_matrix(3, dtype=float)
        self.box_vectors = box_vectors

    def _copy_instance(self) -> Self:
        """
        Make a copy of this RootPrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            box_vectors=self.box_vectors.copy(),
            children=[],
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone
