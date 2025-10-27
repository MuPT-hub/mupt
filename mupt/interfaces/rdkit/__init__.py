'''Interfaces between the hierarchical MuPT molecular representation and RDKit Mol objects'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'


from .selection import (
    # Atom selection
    AtomCondition,
    all_atoms,
    no_atoms,
    atoms_by_condition,
    atom_neighbors_by_condition,
    has_atom_neighbors_by_condition,
    # Bond selection
    BondCondition,
    all_bonds,
    no_bonds,
    bonds_by_condition,
    bond_condition_by_atom_condition_factory,
)
from .components import (
    chemical_graph_from_rdkit,
    atom_positions_from_rdkit,
    connector_between_rdatoms,
    connectors_from_rdkit,
)
from .importers import primitive_from_rdkit
from .exporters import primitive_to_rdkit
from .depiction import (
    set_rdkdraw_size,
    show_substruct_highlights,
    hide_substruct_highlights,
    show_atom_indices,
    hide_atom_indices,
    enable_kekulized_drawing,
    disable_kekulized_drawing,
    clear_highlights,
)

# CORE CHEMISTRY UTILS WHICH ARE RDKIT-SPECIFIC
from ...chemistry.rdloggers import (
    suppress_rdkit_logs,
    RDLoggerNames,
)
from ...chemistry.sanitization import (
    sanitized_mol,
    AROMATICITY_MDL,
    SANITIZE_ALL,
    SANITIZE_NONE,
)

# DEFAULT DRAWING CONFIG
set_rdkdraw_size(400, aspect=3/2)
show_atom_indices()
show_substruct_highlights()
disable_kekulized_drawing()