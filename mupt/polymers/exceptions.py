"""Exceptions raised during polymer topology generation"""


class FaceError(ValueError):
    """Raised when face or end-group specifications are missing or inconsistent"""


class TopologyValidationError(ValueError):
    """Raised when a descriptive graph can't be realized with a given lexicon"""


class CompatibilityError(TopologyValidationError):
    """Raised when the faces at a join of two components can't be matched"""
