"""
Interfaces between the hierarchical MuPT
molecular representation and RDKit Mol objects
"""

from .components import (
    DEFAULT_ATOM_LABELLER as DEFAULT_ATOM_LABELLER,
    DEFAULT_CONNECTOR_LABELLER as DEFAULT_CONNECTOR_LABELLER,
    DEFAULT_ATTACHABLES_FACTORY as DEFAULT_ATTACHABLES_FACTORY,
    chemical_graph_from_rdkit as chemical_graph_from_rdkit,
    atom_positions_from_rdkit as atom_positions_from_rdkit,
    atom_radius_from_rdkit as atom_radius_from_rdkit,
    connector_from_rdkit_atoms as connector_from_rdkit_atoms,
    connector_pair_from_rdkit_bond as connector_pair_from_rdkit_bond,
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

# DEFAULT DRAWING CONFIG
from ...chemistry.rdkit.depiction import (
    set_rdkdraw_size,
    show_substruct_highlights,
    show_atom_indices,
    disable_kekulized_drawing,
)

set_rdkdraw_size(400, aspect=3 / 2)
show_atom_indices()
show_substruct_highlights()
disable_kekulized_drawing()
