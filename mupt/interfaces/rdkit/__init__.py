"""
Interfaces between the hierarchical MuPT
molecular representation and RDKit Mol objects
"""

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
