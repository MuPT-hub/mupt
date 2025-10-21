'''Reference and utilities for Prop attributes on RDKit objects (i.e. Atom, Bond, and Mol objects)'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Any, Callable, Optional, Union

from rdkit.Chem.rdchem import Atom, Bond, Mol, RWMol
RDObj = Union[Atom, Bond, Mol, RWMol]


# REFERENCE FOR "MAGIC" MOL PROP KEYS (https://www.rdkit.org/docs/RDKit_Book.html#romol-mol-in-python)
RDMOL_MAGIC_PROPS = {
    'MolFileComments'        : 'Read from/written to the comment line of CTABs.',
    'MolFileInfo'            : 'Read from/written to the info line of CTABs.',
    '_MolFileChiralFlag'     : 'Read from/written to the chiral flag of CTABs.',
    '_Name'                  : 'Read from/written to the name line of CTABs.',
    '_smilesAtomOutputOrder' : 'The order in which atoms were written to SMILES',
    '_smilesBondOutputOrder' : 'The order in which bonds were written to SMILES',
}
# REFERENCE FOR "MAGIC" ATOM PROP KEYS (https://www.rdkit.org/docs/RDKit_Book.html#atom)
RDATOM_MAGIC_PROPS = {
    '_CIPCode'               : 'the CIP code (R or S) of the atom',
    '_CIPRank'               : 'the integer CIP rank of the atom',
    '_ChiralityPossible'     : 'set if an atom is a possible chiral center',
    '_MolFileRLabel'         : 'integer R group label for an atom, read from/written to CTABs.',
    '_ReactionDegreeChanged' : 'set on an atom in a product template of a reaction if its degree changes in the reaction',
    '_protected'             : 'atoms with this property set will not be considered as matching reactant queries in reactions',
    'dummyLabel'             : '(on dummy atoms) read from/written to CTABs as the atom symbol',
    'molAtomMapNumber'       : 'the atom map number for an atom, read from/written to SMILES and CTABs',
    'molfileAlias'           : 'the mol file alias for an atom (follows A tags), read from/written to CTABs',
    'molFileValue'           : 'the mol file value for an atom (follows V tags), read from/written to CTABs',
    'molFileInversionFlag'   : 'used to flag whether stereochemistry at an atom changes in a reaction, read from/written to CTABs, determined automatically from SMILES',
    'molRxnComponent'        : 'which component of a reaction an atom belongs to, read from/written to CTABs',
    'molRxnRole'             : 'which role an atom plays in a reaction (1=Reactant, 2=Product, 3=Agent), read from/written to CTABs',
    'smilesSymbol'           : 'determines the symbol that will be written to a SMILES for the atom',
}

# REFERENCE TABLES FOR ENFORCING C++ TYPING THAT RDKit ENFORCES
RDPropType = Union[str, int, float, bool]
RDPROP_GETTERS = {
    str   : 'GetProp',
    bool  : 'GetBoolProp',
    int   : 'GetIntProp',
    float : 'GetDoubleProp'
}
RDPROP_SETTERS = {
    str   : 'SetProp',
    bool  : 'SetBoolProp',
    int   : 'SetIntProp',
    float : 'SetDoubleProp'
}

# PROPERTY INSPECTION AND TRANSFER FUNCTIONS
## TODO: implement generic "smart" getters and setter which are type-aware

def isrdobj(obj : Any) -> bool:
    '''Check if the given object is an RDKit object'''
    return isinstance(obj, RDObj.__args__)

def assign_property_to_rdobj(
    rdobj : RDObj,
    prop_name : str,
    prop_value : Any,
    preserve_type : bool=True,
) -> None:
    '''Assign a Python object to a property of an RDKit object in a type-respecting manner'''
    type_setter_name : Optional[str] = RDPROP_SETTERS.get(type(prop_value), None)
    if (type_setter_name is None) or (not preserve_type):
        rdobj.SetProp(prop_name, str(prop_value))
        # TODO: handle listlike props more specifically than this
    else:
        type_setter : Callable[[str, Any], None] = getattr(rdobj, type_setter_name) # DEV: 2nd arg is actually same type as prop_value
        type_setter(prop_name, prop_value)

def copy_rdobj_props(from_rdobj : RDObj, to_rdobj : RDObj) -> None: # NOTE : no need to incorporate typing info, as RDKit objects can correctly interpret typed strings
    '''For copying properties between a pair of RDKit Atoms or Mols'''
    # NOTE : avoid use of GetPropsAsDict() to avoid errors from restrictive C++ typing
    assert isrdobj(from_rdobj) and isrdobj(to_rdobj) # verify that both objects passed are RDKit objects...
    assert type(from_rdobj) == type(to_rdobj)        # ...AND that both objects are the same type of RDKit object

    for prop in from_rdobj.GetPropNames():
        to_rdobj.SetProp(prop, from_rdobj.GetProp(prop))
