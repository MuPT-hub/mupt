"""SimplePrimitive which represent particular atoms of elements on the periodic table"""

from typing import ClassVar, Iterable, Optional, Self

from copy import deepcopy

from .simple import SimplePrimitive
from ..primitive import PrimitiveLabel
from ...connection.connectors import Connector
from ...connection.management import ConnectorManager
from ....geometry.shapes.shapes import BoundedTransformableShape
from ....chemistry.core import ElementLike, isatom, valence_allowed


class AtomicPrimitive(SimplePrimitive):
    """
    A Primitive representing a single atom from the periodic table
    Contains element, formal charge, and nuclear mass information about the atom
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "ATOM"

    def __init__(
        self,
        element: ElementLike,
        connections: Optional[ConnectorManager | Iterable[Connector]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        label: Optional[PrimitiveLabel] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        # TODO: add option to accept element symbol or atomic number
        if not isatom(element):
            raise TypeError(f"Invalid element type {type(element)}")
        self._element = element

        super().__init__(
            connections=connections,
            shape=shape,
            label=label,
            metadata=metadata,
        )

    # Copying
    def _copy_instance(self) -> Self:
        """
        Make a copy of this AtomicPrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            element=self.element,  # TB: double-check this is actually a singleton
            connections=[],  # will be plumbed up in _copy_untransformed()
            shape=None if (self.shape is None) else self.shape.copy(),
            metadata={key: value for key, value in self.metadata.items()},
            label=deepcopy(self.label),
        )
        return clone

    @property  # DEV: no setter implemented; element is immutable after instantiation
    def element(self) -> ElementLike:
        """The chemical element, ion, or isotope associated with this AtomicPrimitive"""
        return self._element

    def check_valence(self) -> None:
        """
        Check that element assigned to atomic Primitives and
        bond orders of Connectors are chemically-compatible
        """
        valence: float = self.connections.valence
        if not valence_allowed(
            self.element.number,
            self.element.charge,
            valence,
        ):
            raise ValueError(
                f"Atomic {self!r} with total valence {valence} "
                "incompatible with assigned element {self.element!r}"
            )
