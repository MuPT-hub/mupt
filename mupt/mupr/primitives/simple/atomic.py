"""SimplePrimitive which represent particular atoms of elements on the periodic table"""

from typing import ClassVar, Iterable, Optional, Self

from .simple import SimplePrimitive
from ..primitive import DEFAULT_ROLE, PrimitiveLabel, Metadata
from ...connection.connectors import Connector
from ...connection.management import ConnectorManager
from ....geometry.shapes.shapes import BoundedTransformableShape
from ....chemistry.core import (
    ELEMENTS as ELEMENTS,
    ElementLike,
    isatom,
    valence_allowed,
)


class AtomicPrimitive(SimplePrimitive):
    """
    A Primitive representing a single atom from the periodic table
    Contains element, formal charge, and nuclear mass information about the atom
    """

    DEFAULT_LABEL: ClassVar[PrimitiveLabel] = "ATOM"

    def __init__(
        self,
        element: ElementLike,
        # TODO: add option to accept element symbol or atomic number
        connections: Optional[ConnectorManager | Iterable[Connector]] = None,
        shape: Optional[BoundedTransformableShape] = None,
        label: Optional[PrimitiveLabel] = None,
        # N.B.: 'mass' deliberately omitted; taken form Element
        role: str = DEFAULT_ROLE,
        metadata: Optional[Metadata] = None,
    ) -> None:
        if not isatom(element):
            raise TypeError(f"Invalid element type {type(element)}")
        self._element = element

        super().__init__(
            connections=connections,
            shape=shape,
            label=label,
            mass=element.mass,
            role=role,
            metadata=metadata,
        )

    def _copy_instance(self) -> Self:
        """
        Make a copy of this AtomicPrimitive WITHOUT any
        hierarchical or Connection information included
        """
        clone = self.__class__(
            element=self.element,  # TODO: double-check this is actually a singleton
            connections=[],  # gets plumbed up in _copy_untransformed()
            **self._copy_attrs_boilerplate(),
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
