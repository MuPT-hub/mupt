"""
Labeled roles that Primitives can play in an exportable hierarchy.

These roles allow Primitives to be explicitly tagged with their semantic
purpose within a molecular representation, enabling generic tree traversal
for export to external toolkits (e.g., MDAnalysis) without
hard-coding assumptions about tree depth or structure.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from enum import Enum
from anytree import PreOrderIter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mupr.primitives import Primitive


class PrimitiveRole(Enum):
    """Labeled roles that Primitives can play in an exportable hierarchy.
    
    These roles map to the standard levels expected by molecular analysis
    toolkits such as MDAnalysis:
    
    - UNASSIGNED: No role has been assigned (default)
    - UNIVERSE:  Root container of the entire system
    - SEGMENT:   Non-covalently bonded entity (chain, molecule)
    - RESIDUE:   Repeating sub-unit (monomer, amino acid, CG bead group)
    - PARTICLE:  Exportable particle (atom in all-atom, bead in CG)

    Primitives at intermediate depths between role-tagged levels should
    use ``UNASSIGNED`` to indicate transparent grouping nodes.
    
    Examples
    --------
    >>> from mupt.mupr.roles import PrimitiveRole
    >>> PrimitiveRole.UNASSIGNED
    <PrimitiveRole.UNASSIGNED: 'unassigned'>
    >>> PrimitiveRole.UNIVERSE is not PrimitiveRole.UNASSIGNED
    True
    """
    UNASSIGNED = "unassigned"
    UNIVERSE = "universe"
    SEGMENT  = "segment"
    RESIDUE  = "residue"
    PARTICLE = "particle"


def has_SAAMR_roles(prim: 'Primitive') -> bool:
    """Check whether a Primitive hierarchy has all four SAAMR roles assigned.

    This checks for role *presence* in the tree, not structural depth.
    A tree passes if at least one node carries each of the four roles:
    UNIVERSE, SEGMENT, RESIDUE, and PARTICLE.

    Parameters
    ----------
    prim : Primitive
        Root of the hierarchy to check.

    Returns
    -------
    bool
        ``True`` if all four SAAMR roles are present in the tree.

    See Also
    --------
    has_strict_SAAMR_depth : Checks for strict depth-3 SAAMR structure.
    assign_SAAMR_roles : Assigns roles to a strict SAAMR hierarchy.
    """
    required = {
        PrimitiveRole.UNIVERSE,
        PrimitiveRole.SEGMENT,
        PrimitiveRole.RESIDUE,
        PrimitiveRole.PARTICLE,
    }
    present = {node.role for node in PreOrderIter(prim)}
    return required.issubset(present)


def assign_SAAMR_roles(prim: 'Primitive') -> None:
    """Assign canonical export roles for a strict depth-3 SAAMR hierarchy.

    Walks the tree by depth (0/1/2/3) and assigns UNIVERSE, SEGMENT,
    RESIDUE, and PARTICLE roles respectively.  Requires that the tree
    satisfies :func:`~mupt.mupr.properties.has_strict_SAAMR_depth`.

    Parameters
    ----------
    prim : Primitive
        Root Primitive of a hierarchy expected to follow SAAMR layout:
        universe -> segment -> residue -> particle (all leaves at depth 3).

    Raises
    ------
    ValueError
        If ``prim`` does not satisfy ``has_strict_SAAMR_depth``.

    See Also
    --------
    has_SAAMR_roles : Checks that roles are already assigned.
    has_strict_SAAMR_depth : The structural precondition for this function.
    """
    from .mupr.properties import has_strict_SAAMR_depth # imported at runtime to avoid circular reference with Primitive

    if not has_strict_SAAMR_depth(prim):
        raise ValueError(
            'Cannot assign SAAMR roles: hierarchy does not have strict '
            'SAAMR depth (all leaves must be atoms at depth 3)'
        )

    prim.role = PrimitiveRole.UNIVERSE

    for segment in prim.children:
        segment.role = PrimitiveRole.SEGMENT

        for residue in segment.children:
            residue.role = PrimitiveRole.RESIDUE

            for particle in residue.children:
                particle.role = PrimitiveRole.PARTICLE
