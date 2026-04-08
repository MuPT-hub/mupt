"""
Properties of Primitives used to assess compatibility with a particular task
E.g. checking atomicity, linearity, adherence to a "standard" hierarchy, etc.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from .primitives import Primitive


def is_SAAMR_compliant(prim: Primitive) -> bool:
   """
   Check whether a Primitive hierarchy is organized
   as universe -> molecule -> repeat-unit -> atom
   
   SAAMR = Standard All-Atom Molecular Representation
   """

   return all(leaf.is_atom and (leaf.depth == 3) for leaf in prim.leaves)
