"""For encoding chemistries and manipulating SMILES-based structures"""

from .core import *
from .linkers import (
    is_linker as is_linker,
    not_linker as not_linker,
    num_linkers as num_linkers,
    anchor_and_linker_idxs as anchor_and_linker_idxs,
)
