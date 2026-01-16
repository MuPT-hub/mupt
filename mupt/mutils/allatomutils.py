'''Utilities related to operations specific to all-atom representations
of molecular systems'''

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

from ..mupr.primitives import Primitive

def _is_AA_export_compliant(prim : Primitive) -> bool:
    """
    Check whether a Primitive hierarchy is organized
    as universe -> chain -> residue -> atom
    """   
    
    return all(
        leaf.is_atom and (leaf.depth == 3)        
            for leaf in prim.leaves
    )