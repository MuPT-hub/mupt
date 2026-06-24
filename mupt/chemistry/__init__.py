'''For encoding chemistries and manipulating SMILES-based structures'''

from .core import *
from .linkers import (
    is_linker,
    not_linker,
    num_linkers,
    anchor_and_linker_idxs,
)