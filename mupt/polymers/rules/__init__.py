"""
Rules for what may bond to what, and how

    faces         : named groups of ports on a unit
    junctions     : how the strands of two meeting faces pair up
    compatibility : which free faces bond when components are joined
"""

from .faces import (
    normalize_faces as normalize_faces,
    validate_faces as validate_faces,
)
from .junctions import (
    JunctionRule as JunctionRule,
    FixedPairing as FixedPairing,
    RandomFlip as RandomFlip,
)
from .compatibility import (
    ANY as ANY,
    CompatibilityRule as CompatibilityRule,
    CompatibleSites as CompatibleSites,
    head_to_tail as head_to_tail,
    free_faces as free_faces,
    resolve_pattern as resolve_pattern,
)
