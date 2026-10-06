"""
Properties of Primitives used to assess compatibility with a particular task
E.g. checking atomicity, linearity, topology, neighbor valence, etc.
"""

from .primitives import (
    Primitive,
    CompositePrimitive,
    AtomicPrimitive,
)


def is_simple(prim: Primitive) -> bool:
    """Whether a Primitive has no internal structure"""
    return prim.is_simple


def supports_children(prim: Primitive) -> bool:
    """Whether a Primitive can have child sub-Primitives"""
    return prim.supports_children


def supports_parents(prim: Primitive) -> bool:
    """Whether a Primitive can be a parent super-Primitive"""
    return prim.supports_parents


def is_atom(prim: Primitive) -> bool:
    """Whether a Primitive represents a single atom from the periodic table"""
    return isinstance(prim, AtomicPrimitive)


def is_superatomic(prim: Primitive) -> bool:
    """
    Whether a Primitive is not itself an atom,
    but all of its DIRECT descendants are
    """
    if not prim.supports_children:
        # specifically exclude atoms themselves, as the empty list of children
        # of an atom would cause the naive all(prim is atom...) to evaluate true
        return False

    for child in prim.children:
        if not is_atom(child):
            return False
    else:
        return True


def is_atomizable(prim: Primitive) -> bool:
    """
    Check whether a Primitive is either an AtomicPrimitive
    or supports children but has only AtomicPrimtive leaves
    """
    # AtomicPrimitives are Simple and therefore must be leaves;
    # no need to recursively check the hierarchy for them
    for leaf in prim.leaves:
        if not is_atom(leaf):
            return False
    else:
        return True


def is_complete(prim: Primitive) -> bool:
    """
    Check whether a Primitive represents a chemically-complete molecular system
    I.e. has no "dangling", unbonded Connectors
    """
    return prim.connections.functionality == 0


def is_flat(prim: CompositePrimitive) -> bool:
    """Check that only one layer exists below the root"""
    return all((leaf.depth == 1) for leaf in prim.leaves)


def is_laminar(prim: CompositePrimitive) -> bool:
    """
    Check that each branch beneath the root has the same depth
    I.e. that children can be arranged in breadth-first
    layer (lamina) traversing down from the root
    """
    seen_depths: set[int] = set()
    for leaf in prim.leaves:
        if not seen_depths:
            seen_depths.add(leaf.depth)
        elif leaf.depth not in seen_depths:
            return False
    else:
        return True


def has_strict_SAAMR_depth(prim: Primitive) -> bool:
    """
    Check whether a Primitive hierarchy is a strict depth-3 SAAMR tree.

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
    for leaf in prim.leaves:
        if not is_atom(leaf) or (leaf.depth != 3):
            return False
    else:
        return True
