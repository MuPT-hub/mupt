"""Fundamental data structures for multiscale molecular representation"""

from .primitive import (
    Primitive as Primitive,
    SupportsParents as SupportsParents,
    SupportsChildren as SupportsChildren,
)
from .root import RootPrimitive as RootPrimitive
from .composite import CompositePrimitive as CompositePrimitive
from .simple import (
    SimplePrimitive as SimplePrimitive,
    AtomicPrimitive as AtomicPrimitive,
)
