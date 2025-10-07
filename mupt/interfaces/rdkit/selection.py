'''Utilities for conditional selection of chemical objects such as atoms and bonds from RDKit molecules'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Concatenate, Generator, Container, Optional, Union
from operator import (
    xor as logical_xor, # alias for consistency
    or_  as logical_or,
    and_ as logical_and,
)
from rdkit.Chem.rdchem import Mol, Bond, Atom


# CONDITIONAL ATOM SELECTION
AtomCondition = Callable[Concatenate[Atom, ...], bool]
AtomLike = Union[int, Atom]

all_atoms : AtomCondition = lambda atom : True
no_atoms  : AtomCondition = lambda atom : False

## ATOM SELECTION FUNCTIONS
def atoms_by_condition(
        mol : Mol,
        condition : Optional[AtomCondition]=None,
        as_indices : bool=False,
        negate : bool=False,
    ) -> Generator[AtomLike, None, None]:
    '''
    Generate a subset of atoms in a Mol based on a condition
    
    Parameters
    ----------
    mol : Chem.Mol
        An RDKit molecule object
    condition : Optional[Callable[[Chem.Atom], bool]], default None
        Condition on atoms which returns bool; 
        Always returns True if unset
    as_indices : bool, default False
        Whether to return results as their indices (default) or as Atom objects
    negate : bool, default False
        Whether to invert the condition provided (by default False)
    
    Returns
    -------
    selected_atoms : Generator[Union[int, Chem.Atom]]
        An iterable Generator of the atoms meeting the chosen condition
    '''
    if condition is None:
        condition = all_atoms
    
    for atom in mol.GetAtoms():
        if logical_xor(condition(atom), negate):
            yield atom.GetIdx() if as_indices else atom
atoms = atoms_by_condition 

def atom_neighbors_by_condition(
        atom : Atom,
        condition : Optional[AtomCondition]=None,
        as_indices : bool=False,
        negate : bool=False,
    ) -> Generator[AtomLike, None, None]:
    '''
    Generate all neighboring atoms (i.e. atoms bonded to the passed atom) satisfying a condition
    
    Parameters
    ----------
    atom : Chem.Atom
        An atom object whose neighbors are to be inspected
    condition : Optional[Callable[[Chem.Atom], bool]], default None
        Condition on atoms which returns bool; 
        Always returns True if unset
    as_indices : bool, default False
        Whether to return results as their indices (default) or as Atom objects
    negate : bool, default False
        Whether to invert the condition provided (by default False)
    
    Returns
    -------
    selected_atoms : Generator[Union[int, Chem.Atom]]
        An iterable Generator of the atoms meeting the chosen condition
    '''
    if condition is None:
        condition = all_atoms

    for nb_atom in atom.GetNeighbors():
        if logical_xor(condition(nb_atom), negate):
            yield nb_atom.GetIdx() if as_indices else nb_atom
atom_neighbors = atom_neighbors_by_condition

def has_atom_neighbors_by_condition(
        atom : Atom,
        condition : Optional[AtomCondition]=None,
        negate : bool=False,
    ) -> bool:
    '''Identify if any neighbors of an atom satisfy some condition'''
    try: 
        next(atom_neighbors_by_condition(atom, condition=condition, negate=negate))
    except StopIteration:
        return False
    else:
        return True
has_atom_neighbors = has_atom_neighbors_by_condition
    

# CONDITIONAL BOND SELECTION
BondCondition = Callable[Concatenate[Bond, ...], bool]
BondLike = Union[int, Bond, tuple[int, int], tuple[Atom, Atom]]

all_bonds : BondCondition = lambda bond : True
no_bonds  : BondCondition = lambda bond : False

## BOND SELECTION FUNCTIONS
def bonds_by_condition(
        mol : Mol,
        condition : Optional[BondCondition]=None,
        as_indices : bool=True,
        as_pairs : bool=True,
        negate : bool=False,
    ) -> Generator[BondLike, None, None]:
    '''
    Select a subset of bonds in a Mol based on a condition
    
    Parameters
    ----------
    mol : Chem.Mol
        An RDKit molecule object
    condition : Optional[Callable[[Chem.Bond], bool]], default None
        Condition on bonds which returns bool; 
        Always returns True if unset
    as_indices : bool, default True
        Whether to return results as Bond objects or their indices (default)
    as_pairs : bool, default True
        Whether to return bonds as the pair of bondss they connect (default) or the bond itself
        Note that if as_pairs=True and as_indices=False, will return as pairs of Bonds objects
    negate : bool, default False
        Whether to invert the condition provided (by default False)
    
    Returns
    -------
    selected_bonds : Generator[Union[int, Bond, tuple[int, int], tuple[Atom, Atom]]]
        A set of the bonds meeting the chosen condition
        Depending on flags set, bond will be represented as:
        * Bond indices
        * Bond objects
        * 2-tuples of Atom objects
        * 2-tuples of atom indices
    '''
    if condition is None:
        condition = all_bonds
    
    for bond in mol.GetBonds():
        if logical_xor(condition(bond), negate):
            if as_pairs:
                yield (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) if as_indices else (bond.GetBeginAtom(), bond.GetEndAtom())
            else:
                yield bond.GetIdx() if as_indices else bond
bonds = bonds_by_condition

def bond_condition_by_atom_condition_factory(
        atom_condition : AtomCondition,
        binary_operator : Callable[[bool, bool], bool]=logical_or,
    ) -> BondCondition:
    '''
    Dynamically define a bond condition based on an atom condition applied to the pair of atom a bond connects
    
    Evaluation over bond determined by a specified atom condition and a binary logical comparison made between the pair of atom condition evaluations
    By default, this binary condition is OR (i.e. the bond will evaluate True if either of its atoms meets the atom condition)
    '''
    def bond_condition(bond : Bond) -> bool:
        return binary_operator(
            atom_condition(bond.GetBeginAtom()),
            atom_condition(bond.GetEndAtom()),
        )
    return bond_condition


# QUERIES BY PREDEFINED CONDITIONS
atom_is_mapped : AtomCondition = lambda atom : atom.GetAtomMapNum() != 0
atom_is_linker : AtomCondition = lambda atom : atom.GetAtomicNum() == 0

def mapped_atoms(mol : Mol, as_indices : bool=False) -> Generator[AtomLike, None, None]:
    '''Return all atoms (either as Atom objects or as indices) which have been assigned a nonzero atom map number'''
    return atoms_by_condition(
        mol,
        condition=atom_is_mapped,
        as_indices=as_indices,
        negate=False,
    )

def mapped_neighbors(atom : Atom, as_indices : bool=False) -> Generator[AtomLike, None, None]:
    '''Return all mapped atoms that an atom is bonded to'''
    return atom_neighbors_by_condition(
        atom,
        condition=atom_is_mapped,
        as_indices=as_indices,
        negate=False,
    )

def bonded_pairs(mol : Mol, *atom_idxs : Container[int], as_indices : bool=True, as_pairs : bool=True) -> Generator[BondLike, None, None]:
    '''Returns all bonds in a Mol which connect a pair of atoms whose indices both lie within the given atom indices'''
    return bonds_by_condition(
        mol,
        condition=bond_condition_by_atom_condition_factory(
            atom_condition=lambda atom : atom.GetIdx() in atom_idxs,
            binary_operator=logical_and,
        ),
        as_indices=as_indices,
        as_pairs=as_pairs,
        negate=False, # NOTE: negate doesn't behave exactly as one might expect here due to de Morgan's laws (i.e. ~(A^B) != (~A^~B))
    )
    
def bonds_between_mapped_atoms(mol : Mol, as_indices : bool=True, as_pairs : bool=True) -> Generator[BondLike, None, None]:
    '''Returns all bonds spanning between two mapped (i.e. nonzero atom map number) atoms'''
    return bonds_by_condition(
        mol,
        condition=bond_condition_by_atom_condition_factory(
            atom_condition=atom_is_mapped,
            binary_operator=logical_and, # only return bond when BOTH atoms are unmapped
        ),
        as_indices=as_indices,
        as_pairs=as_pairs,
        negate=False, # NOTE: negate doesn't behave exactly as one might expect here due to de Morgan's laws (i.e. ~(A^B) != (~A^~B))
    )
