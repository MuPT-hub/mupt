"""MDAnalysis interface for MUPT."""

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'


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

# DEFAULT DRAWING CONFIG
set_rdkdraw_size(400, aspect=3/2)
show_atom_indices()
show_substruct_highlights()
disable_kekulized_drawing()


from .exporters import primitive_to_mdanalysis

