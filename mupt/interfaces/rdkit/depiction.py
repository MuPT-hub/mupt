'''Drawing config for RDKit molecule depiction'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Draw import IPythonConsole


# GLOBAL PREFERENCES
def set_rdkdraw_size(dim : int=300, aspect : float=3/2):
    '''Change image size and shape of RDKit Mol images'''
    IPythonConsole.molSize = (int(aspect*dim), dim) # Change IPython image display size
    
def show_substruct_highlights() -> None:
    '''Turns on highlighting of found substructures when performing substructure matches'''
    IPythonConsole.highlightSubstructs = True

def hide_substruct_highlights() -> None:
    '''Turns off highlighting of found substructures when performing substructure matches'''
    IPythonConsole.highlightSubstructs = False
    
def show_atom_indices() -> None:
    '''Turns on atom index display when drawing molecules in Jupyter Notebooks'''
    IPythonConsole.drawOptions.addAtomIndices = True
    
def hide_atom_indices() -> None:
    '''Turns off atom index display when drawing molecules in Jupyter Notebooks'''
    IPythonConsole.drawOptions.addAtomIndices = False

def enable_kekulized_drawing() -> None:
    '''Turns on automatic kekulization of aromatic bonds before drawing molecules in Jupyter Notebooks'''
    IPythonConsole.kekulizeStructures = True

def disable_kekulized_drawing() -> None:
    '''Turns off automatic kekulization of aromatic bonds before drawing molecules in Jupyter Notebooks'''
    IPythonConsole.kekulizeStructures = False

# DISPLAY OPTIONS FOR INDIVIDUAL MOLECULES
def clear_highlights(rdmol : Mol) -> None:
    '''Removes the highlighted atoms flags from an RDKit Mol if present'''
    if hasattr(rdmol, '__sssAtoms'):
        del rdmol.__sssAtoms