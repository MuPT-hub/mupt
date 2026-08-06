'''mBuild interface: export a MuPT Primitive hierarchy to an mBuild Compound.

This subpackage is optional and is only loaded on explicit import, so mBuild is
not a hard dependency of importing mupt. Import it directly:

    from mupt.interfaces.mbuild import to_compound
'''

from .._shared.topology import (
    ResolutionPredicate,
    ResolutionSpec,
    resolution_predicate,
    collect_resolution_nodes,
    resolution_graph,
)
from .exporters import (
    to_compound,
    to_gmso,
)

__all__ = [
    'ResolutionPredicate',
    'ResolutionSpec',
    'resolution_predicate',
    'collect_resolution_nodes',
    'resolution_graph',
    'to_compound',
    'to_gmso',
]
