"""
Interfaces between the hierarchical MuPT
molecular representation and RDKit Mol objects
"""

## TB DEV: "from Y import X as X" looks pretty silly and redundant,
## but skirts around the "unused variable" flag otherwise raised by linter
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
from .components import (
    chemical_graph_from_rdkit as chemical_graph_from_rdkit,
    atom_positions_from_rdkit as atom_positions_from_rdkit,
    connector_between_rdatoms as connector_between_rdatoms,
    connectors_from_rdkit as connectors_from_rdkit,
)
from .importers import primitive_from_rdkit as primitive_from_rdkit
from .exporters import (
    primitive_to_rdkit as primitive_to_rdkit,
    primitive_to_rdkit_mols as primitive_to_rdkit_mols,
)
from .strategies import (
    RDKitExportStrategy as RDKitExportStrategy,
    AllAtomRDKitExportStrategy as AllAtomRDKitExportStrategy,
)
from .depiction import (
    set_rdkdraw_size,
    show_substruct_highlights,
    hide_substruct_highlights as hide_substruct_highlights,
    show_atom_indices,
    hide_atom_indices as hide_atom_indices,
    enable_kekulized_drawing as enable_kekulized_drawing,
    disable_kekulized_drawing,
    clear_highlights as clear_highlights,
)

# CORE CHEMISTRY UTILS WHICH ARE RDKIT-SPECIFIC
from ...chemistry.rdloggers import (
    suppress_rdkit_logs as suppress_rdkit_logs,
    RDLoggerNames as RDLoggerNames,
)
from ...chemistry.sanitization import (
    sanitized_mol as sanitized_mol,
    AROMATICITY_MDL as AROMATICITY_MDL,
    SANITIZE_ALL as SANITIZE_ALL,
    SANITIZE_NONE as SANITIZE_NONE,
)

# DEFAULT DRAWING CONFIG
set_rdkdraw_size(400, aspect=3 / 2)
show_atom_indices()
show_substruct_highlights()
disable_kekulized_drawing()
