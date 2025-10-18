'''Reference for fundamental chemical units, namely elements, ions, isotopes, and bond types'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Union

from rdkit.Chem.rdchem import Atom, BondType, GetPeriodicTable
RDKitPeriodicTable = GetPeriodicTable()

from periodictable import elements
from periodictable.core import Element, Ion, Isotope, isatom
ELEMENTS = elements
ElementLike = Union[Element, Ion, Isotope]


def valence_allowed(atomic_num : int, charge : int, valence : int) -> bool:
    '''Check if the given valence is allowed for the specified element'''
    ## Calculation based on RDKit's valence prescription (https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences)
    ## ..., down to the treatment of charged atoms by their isoelectronic equivalents
    effective_atomic_num = atomic_num - charge # e.g. treat [N+] as C, [N-] as O, etc.
    allowed_valences = RDKitPeriodicTable.GetValenceList(effective_atomic_num)
    
    if -1 in allowed_valences:
        return True
    return valence in allowed_valences # TODO: write unit tests


