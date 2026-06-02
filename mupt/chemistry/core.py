'''Reference for fundamental chemical units, namely elements, ions, isotopes, and bond types'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from typing import Union

from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdchem import Atom, BondType, GetPeriodicTable
RDKitPeriodicTable = GetPeriodicTable()

from periodictable import elements
from periodictable.core import Element, Ion, Isotope, isatom
ELEMENTS = elements
ElementLike = Union[Element, Ion, Isotope]

from .rdloggers import suppress_rdkit_logs


def _compile_bond_order_reference() -> dict[BondType, float]:
    '''
    Generate reference table of BondType to corresponding electronic bond order 
    (e.g. aromatic = 1.5, double = 2, etc.), consistent with RDKit's definition
    '''
    dummy = MolFromSmiles('*-*')
    bond = dummy.GetBondWithIdx(0) # DEV: can't directly initialize Bond from Python, so using this hacky aprroach to setup instead

    bond_orders_by_bond_type : dict[BondType, float] = dict()
    for bondtype in BondType.names.values():
        bond.SetBondType(bondtype)
        with suppress_rdkit_logs('rdApp.error'):
            try:
                # N.B.: these values are NOT the same as the keys of BondType.values; those are arbitrary indices, 
                # whereas the bond order here conveys info loosely about the number of electrons per bond
                bond_orders_by_bond_type[bondtype] = bond.GetBondTypeAsDouble()
            except RuntimeError:
                # DEV: functions as a warning, but want this to be suppressed nominally
                LOGGER.debug(f'RDKit BondType {bondtype!s} does not have a double-valued bond order defined')

    return bond_orders_by_bond_type
BOND_ORDER : dict[BondType, float] = _compile_bond_order_reference()

def valence_allowed(atomic_num : int, charge : int, valence : int) -> bool:
    '''Check if the given valence is allowed for the specified element'''
    if atomic_num == 0:
        return True # skip checks for linkers (should not be interpreted as neutrons, which they would be if passed thru the logic below)

    ## Calculation based on RDKit's valence prescription (https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences)
    ## ..., down to the treatment of charged atoms by their isoelectronic equivalents
    effective_atomic_num = atomic_num - charge # e.g. treat [N+] as C, [N-] as O, etc.
    allowed_valences = RDKitPeriodicTable.GetValenceList(effective_atomic_num)
    
    if -1 in allowed_valences:
        return True
    return valence in allowed_valences # TODO: write unit tests


