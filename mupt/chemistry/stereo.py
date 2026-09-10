"""Utilities for handling stereochemistry, including CIP assignment and enumeration of stereoisomers"""


# DEVNOTE: just doing a kitchen sink import for now so I remember later what all RDKit has to offer here
# for comprehensive documentation, see https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops

STEREOINFO_ATTRS: tuple[str] = (
    "NOATOM",
    "centeredOn",
    "controllingAtoms",
    "descriptor",
    "permutation",
    "specified",
    "type",
)
