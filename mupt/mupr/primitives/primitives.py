"""
Base classes for Primitives, the fundamental datatype
within MuPT's multiscale molecular representation
"""

from typing import Any, ClassVar, Collection, Hashable, Optional

type PrimitiveLabel = Hashable
type PrimitiveAddress = Hashable

from anytree.node import NodeMixin

from ..connection.connectors import Connector
from ..connection.management import ConnectorManager

from ...mutils.referencing import Addressed
from ...mutils.containers import Labelled
from ...geometry.shapes import Shaped, BoundedTransformableShape
from ...geometry.transforms.rigid import RigidlyTransformable


class Primitive(
    Addressed,  # N.B.: each subtype will have its own class-wide registry
    Labelled,
    Shaped,
    RigidlyTransformable,
    NodeMixin,
):
    """A fundamental, scale-agnostic building block of a molecular system"""

    # Class attributes
    ## Global
    DEFAULT_LABEL: ClassVar[PrimitiveLabel]
    ADDRESS_PREVIEW_LEN: ClassVar[int] = 7

    ## Subtype-specific
    # TB: considered dropping the Law of the Excluded Middle here, i.e. via an
    # Optional[bool] type annotation, since strictly-speaking these properties are
    # UNDEFINED on the base, being neither true nor false
    # Opting not to now for type simplicity, since None is Falsy anyways
    IS_SIMPLE: ClassVar[bool] = False
    SUPPORTS_CHILDREN: ClassVar[bool] = False
    SUPPORTS_PARENTS: ClassVar[bool] = False

    # TB: am slightly uneasy of working around class hierarchy inversion this way,
    # but does definitely make preconditions and properties over Primitives much simpler
    @property
    def is_simple(self) -> bool:
        """Whether this Primitives is considered indivisible within its hierarchy"""
        # DEV: this is a mechanism to prevent Simples from being the parents of any
        # other Primitive without passing type info backward up the inheritance tree
        return self.IS_SIMPLE

    @property
    def supports_children(self) -> bool:
        """
        Whether this Primitive is capable of having other
        Primitives below it (i.e. at a "finer" resolution)
        in the MuPT molecular representation hierarchy
        """
        return self.SUPPORTS_CHILDREN

    @property
    def supports_parents(self) -> bool:
        """
        Whether this Primitive is capable of being bound as the child of
        another "parent" Primitive (which is at a "coarser" resolution)
        in the MuPT molecular representation hierarchy
        """
        return self.SUPPORTS_PARENTS

    # Instance attributes
    connections: ConnectorManager
    _shape: Optional[BoundedTransformableShape]  # TODO: add protected access
    _label: PrimitiveLabel
    metadata: dict[Hashable, Any]

    _frozen_connections: bool
    _frozen_hierarchy: bool

    # Creation
    ## N.B.: deliberately not placeing __init__ method here;
    ## should NOT be able to create instance of Primitive base
    ## TB: given that, may consider pivoting from Protocol -> ABC?
    def _init(
        self,
        # DEV: connectors deliberately omitted here; subtype each handle in unique way
        shape: Optional[BoundedTransformableShape] = None,
        label: Optional[PrimitiveLabel] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Boilerplate initialization of hidden attrs, with defaults
        To be called inside __init__ of concrete subtypes
        """
        # hidden flags - mutable by default
        self._frozen_connections = False
        self._frozen_hierarchy = False

        # instance attrs
        self._shape = shape
        self.metadata = metadata or dict()
        self._label = label

    @property
    def label(self) -> PrimitiveLabel:
        """
        A distinguishing label which can be assigned
        by the user for identification purposes
        """
        if self._label is None:
            # Attempt fallbacks to fetch label
            if "label" in self.metadata:
                self._label = self.metadata["label"]
            else:
                self._label = self.DEFAULT_LABEL

        return self._label

    @label.setter
    def label(self, new_label: Optional[PrimitiveLabel]) -> None:
        """Assign a new label to this Primitive"""
        if new_label is None:
            new_label = self.DEFAULT_LABEL
        self._label = new_label

    # Wrapped properties
    @property
    def connectors(self) -> Collection[Connector]:
        """Convenience wrapper for accessing ALL connectors managed by this Primitive"""
        # TODO: also provide convenient access to connectors_free and connectors_bound
        return self.connections.connectors

    @property
    def functionality(self) -> int:
        """Number of free Connections this Primitive has access to"""
        return self.connections.functionality
