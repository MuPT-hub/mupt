'''
Properties of Primitives used to assess compatibility with a particular task
E.g. checking atomicity, linearity, topology, neighbor valence, etc.
'''

__author__ = 'Timotej Bernat, Joseph R. Laforet Jr.'
__email__ = 'timotej.bernat@colorado.edu, jola3134@colorado.edu'

from .primitives import (
    Primitive,
    SimplePrimitive,
    AtomicPrimitive,
    CompositePrimitive,
)

def is_simple(prim : Primitive) -> bool:
    '''Check whether a Primitive has no internal structure'''
    return isinstance(prim, SimplePrimitive)

def is_atom(prim : Primitive) -> bool:
    '''Check whether a Primitive represents a single atom from the periodic table'''
    return isinstance(prim, AtomicPrimitive)

def is_atomizable(prim : Primitive) -> bool:
    '''Check whether a Primitive is either an AtomicPrimitive or a CompositePrimitive which can be fully expanded into AtomicPrimitives'''
    if is_atom(prim):
        return True
    
    if not isinstance(prim, CompositePrimitive):
        return False
    
    return all(
        is_atomizable(child)
            for child in prim.children
    )

def is_complete(prim : Primitive) -> bool:
    '''
    Check whether a Primitive represents a chemically-complete molecular system
    I.e. has no "dangling", unbonded Connectors
    '''
    if prim.functionality > 0: # no unsaturated connections
        return False
    
    if isinstance(prim, SimplePrimitive):
        return True
    elif isinstance(prim, CompositePrimitive): # children might still have free connectors
        return all( # no dangling composites
            is_complete(child)
                for child in prim.children
        )
    else:
        raise TypeError

def is_flat(prim : CompositePrimitive) -> bool:
    '''Check that only one layer exists below the root'''
    return all(
        (leaf.depth == 1)
            for leaf in prim.leaves
    )

def is_laminar(prim : CompositePrimitive) -> bool:
    '''
    Check that each branch beneath the root has the same depth
    I.e. that children can be arranged in breadth-first layers (lamina) traversing down from the root
    '''
    seen_depths : set[int] = set()
    for leaf in prim.leaves:
        if not seen_depths:
            seen_depths.add(leaf.depth)
        elif leaf.depth not in seen_depths:
            return False
    else: return True


def has_strict_SAAMR_depth(prim: Primitive) -> bool:
    '''
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
    '''
    # TODO: type-check for Composites
    return all(
        leaf.is_atom and (leaf.depth == 3)
        for leaf in prim.leaves
    )
