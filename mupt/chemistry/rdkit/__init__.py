"""Chemical utilities specific to RDKit"""

## TB DEV: "from Y import X as X" looks pretty silly and redundant,
## but skirts around the "unused variable" flag otherwise raised by linter
from .sanitization import (
    sanitized_mol as sanitized_mol,
    AROMATICITY_MDL as AROMATICITY_MDL,
    SANITIZE_ALL as SANITIZE_ALL,
    SANITIZE_NONE as SANITIZE_NONE,
)
from .selection import (
    # Atom selection
    AtomCondition as AtomCondition,
    all_atoms as all_atoms,
    no_atoms as no_atoms,
    atoms_by_condition as atoms_by_condition,
    atom_neighbors_by_condition as atom_neighbors_by_condition,
    has_atom_neighbors_by_condition as has_atom_neighbors_by_condition,
    # Bond selection
    BondCondition as BondCondition,
    all_bonds as all_bonds,
    no_bonds as no_bonds,
    bonds_by_condition as bonds_by_condition,
)
from .linkers import (
    is_linker as is_linker,
    not_linker as not_linker,
    num_linkers as num_linkers,
    anchor_and_linker_idxs as anchor_and_linker_idxs,
)

from .depiction import (
    set_rdkdraw_size as set_rdkdraw_size,
    show_substruct_highlights as show_substruct_highlights,
    hide_substruct_highlights as hide_substruct_highlights,
    show_atom_indices as show_atom_indices,
    hide_atom_indices as hide_atom_indices,
    enable_kekulized_drawing as enable_kekulized_drawing,
    disable_kekulized_drawing as disable_kekulized_drawing,
    clear_highlights as clear_highlights,
)
from .rdloggers import (
    suppress_rdkit_logs as suppress_rdkit_logs,
    RDLoggerNames as RDLoggerNames,
)
