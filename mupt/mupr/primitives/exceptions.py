"""Exceptions specific to Primitives and related operations"""


class ImproperHierarchyError(AttributeError):
    """
    Raised when attempting to use a sybtype of Primtiive in a
    place where it can't be used in a hierarchical representation
    """

    ...


class ArborescenceError(ImproperHierarchyError):
    """Raised when trying to use a Root as the child of another Primitive"""

    ...


class IrreducibilityError(ImproperHierarchyError):
    """
    Raised when attempting to perform a
    composite Primitive operation on a simple one
    """

    ...
