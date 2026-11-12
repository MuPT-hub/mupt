"""
mBuild interface: export a MuPT Primitive hierarchy to an mBuild Compound,
and read updated Compound positions back onto that hierarchy.
This subpackage is optional and is only loaded on explicit import,
so mBuild is not a hard dependency of importing mupt. Import it directly:
"""

from .exporters import primitive_to_mbuild as primitive_to_mbuild
from .importers import update_primitive_from_mbuild as update_primitive_from_mbuild
