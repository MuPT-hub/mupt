'''Perception and manipulation of wild-type "linker" atoms and their in an RDKit molecule'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Generator

from rdkit import Chem
from rdkit.Chem import Atom, Bond, Mol


# DEVNOTE: unclear whether X (total connections) or D (explicit connections) is the right choice for this query...
# ...or if there's ever a case where the two would not produce identical results; both seem to handle higher-order bonds correctly (i.e. treat double bond as "one" connection)
# LINKER_QUERY = '[#0D1]~[!#0]' # neutronium-excluding linker query; requires that the linker be attached to a non-linker atom
LINKER_QUERY : str = '[#0X1]~*' # atomic number 0 (wild) attached to exactly 1 of anything (including possibly another wild-type atom)
LINKER_QUERY_MOL : Mol = Chem.MolFromSmarts(LINKER_QUERY)

def is_linker(rdatom : Atom) -> bool:
    '''Indicate whether an atom is a linker (intermonomer "*" type atom)'''
    return rdatom.GetAtomicNum() == 0

def not_linker(rdatom : Atom) -> bool:
    '''Indicate whether an atom is NOT a linker, i.e. is a "real" atom'''
    # return rdatom.GetAtomicNum() != 0
    return not is_linker(rdatom)
is_real_atom = not_linker

def get_num_linkers(rdmol : Mol) -> int:
    '''Count how many wild-type inter-molecule linker atoms are in a Mol'''
    return sum(
        is_linker(atom)
            for atom in rdmol.GetAtoms()
    )
    
def get_linker_and_bridgehead_idxs(rdmol : Mol) -> Generator[tuple[int, int], None, None]:
    '''Get the linker and bridgehead indices of all ports found in an RDKit Mol'''
    # DON'T de-duplify (i.e. uniquify) indices of substruct matches (fails to catch both ports on a neutronium)
    for (linker_id, bh_id) in rdmol.GetSubstructMatches(LINKER_QUERY_MOL, uniquify=False):
        yield linker_id, bh_id # unpacked purely for self-documentation

def renumber_linkers_as_last(rdmol : Mol) -> Mol: # TODO: make optionally in-place
    '''
    Returns a copy of a Mol whose atom indices are renumbered such that:
    * all #L linker atoms are assigned the last L indices (i.e. occur after all real atoms in order)
    * all non-linker (i.e. "real") atom are numbered in the order they appear in the original Mol
    '''
    linker_idxs : list[int] = []
    real_atom_idxs : list[int] = []
    for atom in rdmol.GetAtoms():
        (real_atom_idxs, linker_idxs)[is_linker(atom)].append(atom.GetIdx())
        
    return Chem.RenumberAtoms(rdmol, real_atom_idxs + linker_idxs)