"""
Logical predicates on Primitives, which evaluate whether a given
Primitive has a specified property as either 'True' or 'False'

Most commonly used for search, filtering, and selecting a working level of resolution
"""

from typing import Callable

from .primitive import Primitive
from .root import RootPrimitive
from .composite import CompositePrimitive

# from .simple.simple import SimplePrimitive
from .simple.atomic import AtomicPrimitive

from .primitives_legacy import PrimitiveLegacy


# Subtype-based properties
# TB: some of these may be suited to be method of Primitive, w/ wrapped calls here
def is_simple(prim: Primitive) -> bool:
    """Whether a Primitive has no internal structure"""
    return prim.is_simple


def supports_children(prim: Primitive) -> bool:
    """Whether a Primitive can have child sub-Primitives"""
    return prim.supports_children


def supports_parents(prim: Primitive) -> bool:
    """Whether a Primitive can be a parent super-Primitive"""
    return prim.supports_parents


def subcomposite(prim: Primitive) -> bool:
    """Whether this Primitive's parent is a CompositePrimitive"""
    return isinstance(prim.parent, CompositePrimitive)


def subroot(prim: Primitive) -> bool:
    """Whether this Primitive's parent is a RootPrimitive"""
    return isinstance(prim.parent, RootPrimitive)


# Hierarchy properties
def has_depth_factory(depth: int) -> Callable[["Primitive"], bool]:
    """Create predicates for whether a Primitive has a particular depth"""

    def has_depth(prim: Primitive) -> bool:
        return prim.depth == depth

    has_depth.__doc__ = f"""
    Whether a Primitive is {depth} levels below the topmost
    Primitive in its hierarchy (not neccesarily a RootPrimitive)"""

    return has_depth


is_dominant = has_depth_factory(0)
is_subdominant = has_depth_factory(1)


def is_complete(prim: Primitive) -> bool:
    """
    Check whether a Primitive represents a chemically-complete molecular system
    I.e. has no "dangling", unbonded Connectors
    """
    return prim.connections.functionality == 0


def is_flat(prim: Primitive) -> bool:
    """Check that only one layer exists below the root"""
    if prim.is_simple:
        return False

    return all((leaf.depth == 1) for leaf in prim.leaves)


def is_laminar(prim: Primitive) -> bool:
    """
    Check that each branch beneath the root has the same depth
    I.e. that children can be arranged in breadth-first
    layer (lamina) traversing down from the root
    """
    if prim.is_simple:
        # Vacuously true; if there are no branches, then all branches are the same depth
        return True

    seen_depths: set[int] = set()
    for leaf in prim.leaves:
        if not seen_depths:
            seen_depths.add(leaf.depth)
        elif leaf.depth not in seen_depths:
            return False
    else:
        return True


# Atom-based properties
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


# SAAMR properties
# TB: TODO (for anyone) - switch typehint to JUST Primitive, once downstream
# functionality depending on PrimitiveLegacy has been refactored
def has_strict_SAAMR_depth(prim: "PrimitiveLegacy | Primitive") -> bool:
    """Check whether a Primitive hierarchy is a strict depth-3 SAAMR tree.

    A strict SAAMR tree has exactly four levels:
    universe (depth 0) -> segment (depth 1) -> residue (depth 2)
    -> particle (depth 3), with every leaf being an atom at depth 3.

    SAAMR = Standard All-Atom Molecular Representation

    This is the structural precondition required by
    :func:`~mupt.mupr.primitives.roles.assign_SAAMR_roles`, which walks the tree
    by depth to assign roles.  MDAnalysis export itself does **not**
    require strict depth-3 structure — any tree with the four SAAMR
    roles assigned can be exported regardless of depth.  Use
    :func:`~mupt.mupr.primitives.roles.has_SAAMR_roles` to check role presence
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
    # is_atom is a predicate function for Primitive, but a property on PrimitiveLegacy
    return all(
        (is_atom(leaf) or getattr(leaf, "is_atom", False) is True) and (leaf.depth == 3)
        for leaf in prim.leaves
    )
