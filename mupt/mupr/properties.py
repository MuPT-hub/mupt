"""
Properties of Primitives used to assess compatibility with a particular task
E.g. checking atomicity, linearity, adherence to a "standard" hierarchy, etc.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from .primitives import Primitive


def has_strict_SAAMR_depth(prim: Primitive) -> bool:
    """Check whether a Primitive hierarchy is a strict depth-3 SAAMR tree.

    A strict SAAMR tree has exactly four levels:
    universe (depth 0) -> segment (depth 1) -> residue (depth 2)
    -> particle (depth 3), with every leaf being an atom at depth 3.

    SAAMR = Standard All-Atom Molecular Representation

    This is the structural precondition required by
    :func:`~mupt.mupr.roles.assign_SAAMR_roles`, which walks the tree
    by depth to assign roles.  MDAnalysis export itself does **not**
    require strict depth-3 structure — any tree with the four SAAMR
    roles assigned can be exported regardless of depth.  Use
    :func:`~mupt.mupr.roles.has_SAAMR_roles` to check role presence
    instead.

    Parameters
    ----------
    prim : Primitive
        Root of the hierarchy to check.

    Returns
    -------
    bool
        ``True`` if every leaf is an atom at depth exactly 3.

    See Also
    --------
    has_SAAMR_roles : Checks that all four SAAMR roles are present (any depth).
    assign_SAAMR_roles : Assigns roles to a strict SAAMR hierarchy.
    """
    return all(leaf.is_atom and (leaf.depth == 3) for leaf in prim.leaves)
