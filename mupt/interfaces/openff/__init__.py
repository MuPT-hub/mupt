"""OpenFF Toolkit interface for MuPT.

This module provides utilities for converting MuPT Primitive hierarchies
to OpenFF Molecule objects and serializing/deserializing OpenFF systems.
"""

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'


from .exporters import (
    primitive_to_openff_molecules,
    primitive_to_openff_topology,
)
from .serialization import (
    save_openff_system,
    load_openff_system,
    save_openff_molecule,
    load_openff_molecule,
)
