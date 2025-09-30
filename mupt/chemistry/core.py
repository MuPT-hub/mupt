'''Reference for fundamental chemical units, namely elements, ions, isotopes, and bond types'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Union

from rdkit.Chem.rdchem import BondType, GetPeriodicTable
RDKitPeriodicTable = GetPeriodicTable()

from periodictable import elements
ELEMENTS = elements

from periodictable.core import Element, Ion, Isotope, isatom
ElementLike = Union[Element, Ion, Isotope]

