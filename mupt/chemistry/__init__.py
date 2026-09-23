"""For encoding chemistries and manipulating SMILES-based structures"""

from .core import (
    Element as Element,
    Ion as Ion,
    Isotope as Isotope,
    ElementLike as ElementLike,
    ELEMENTS as ELEMENTS,
    BOND_ORDER as BOND_ORDER,
    RDKitPeriodicTable as RDKitPeriodicTable,
    valence_allowed as valence_allowed,
)
from .linkers import (
    is_linker as is_linker,
    not_linker as not_linker,
    num_linkers as num_linkers,
    anchor_and_linker_idxs as anchor_and_linker_idxs,
)
