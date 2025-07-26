'''Utilities for handling stereochemistry, including CIP assignment and enumeration of stereoisomers'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from rdkit.Chem.rdchem import (
    StereoInfo,
    StereoType,
    StereoDescriptor,
    ChiralType,
)
# DEVNOTE: just doing a kitchen sink import for now so I remember later what all RDKit has to offer here
# for comprehensive documentation, see https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops
from rdkit.Chem.rdmolops import ( 
    AssignStereochemistry,
    AssignStereochemistryFrom3D,
    AddStereoAnnotations,
    AssignChiralTypesFromBondDirs,
    AssignAtomChiralTagsFromStructure,
    AssignAtomChiralTagsFromMolParity,
    FindPotentialStereo,
    FindPotentialStereoBonds,
)

STEREOINFO_ATTRS : tuple[str] = (
    'NOATOM',
    'centeredOn',
    'controllingAtoms',
    'descriptor',
    'permutation',
    'specified',
    'type'
)