"""
Base classes for Primitives, the fundamental datatype
within MuPT's multiscale molecular representation
"""

from typing import Any, ClassVar, Hashable, Optional

type PrimitiveLabel = Hashable
type PrimitiveAddress = Hashable

from anytree.node import NodeMixin

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

    # Instance attributes
    connections: ConnectorManager
    _shape: Optional[BoundedTransformableShape]  # TODO: add protected access
    _label: PrimitiveLabel
    metadata: dict[Hashable, Any]

    _frozen_connections: bool
    _frozen_hierarchy: bool
