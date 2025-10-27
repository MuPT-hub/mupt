'''For encoding chemistries and manipulating SMILES-based structures'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from .core import *
from .linkers import (
    is_linker,
    not_linker,
    num_linkers,
    anchor_and_linker_idxs,
)