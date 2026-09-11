"""
Utilities for handling stereochemistry, including 
CIP assignment and enumeration of stereoisomers
"""

# TB DEVNOTE: just doing a kitchen sink import for now so I remember
# later what RDKit has to offer here for comprehensive documentation.
# See https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops

STEREOINFO_ATTRS: tuple[str, ...] = (
    "NOATOM",
    "centeredOn",
    "controllingAtoms",
    "descriptor",
    "permutation",
    "specified",
    "type",
)
