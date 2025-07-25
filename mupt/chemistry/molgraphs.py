'''Operations for generating, parsing, and isomorphism searches on molecular graphs'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import Callable, Hashable, Optional

import networkx as nx
from rdkit import Atom, Mol

from .selection import (
    AtomCondition,
    BondCondition,
    logical_or,
    all_atoms,
    bonds_by_condition,
    bond_condition_by_atom_condition_factory
)


def chemical_graph_from_rdkit(
    rdmol : Mol,
    atom_condition : AtomCondition=None,
    label_method : Callable[[Atom], Hashable]=lambda atom : atom.GetIdx(),
    binary_operator : Callable[[bool, bool], bool]=logical_or,
) -> nx.Graph:
    '''
    Create a graph from an RDKit Mol whose:
    * Vertices correspond to all atoms satisfying the given atom condition, and
    * Edges are all bonds between the selected atoms
    
    Parameters
    ----------
    rdmol : Chem.Mol
        The RDKit Mol object to convert.
    atom_condition : Optional[Callable[[Chem.Atom], bool]], default None
        Condition on atoms which returns bool; 
        Always returns True if unset
    label_method : Callable[[Chem.Atom], Hashable], default lambda atom : atom.GetIdx()
        Method to uniquely label each atom as a vertex in the graph
        Default to choosing the atom's index
    binary_operator : Callable[[bool, bool], bool], default logical_or
        Binary logical operator used to 
    '''
    if not atom_condition:
        atom_condition : AtomCondition = all_atoms
    bond_condition : BondCondition = bond_condition_by_atom_condition_factory(atom_condition, binary_operator)
    
    return nx.Graph(
        (label_method(atom_begin), label_method(atom_end))
            for (atom_begin, atom_end) in bonds_by_condition(
                rdmol,
                condition=bond_condition,
                as_pairs=True,      # return bond as pair of atoms,
                as_indices=False,   # ...each as Atom objects
                negate=False,
            )
    )