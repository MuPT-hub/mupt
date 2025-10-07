'''
Utilities for extracting information from and recasting RDKit objects
(e.g. Atom, Bond, Conformer, etc.) and recasting them as MuPT core objects
'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'


from typing import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Optional,
    Type,
    TypeVar,
)

import numpy as np
import networkx as nx
GraphLike = TypeVar('GraphLike', bound=nx.Graph)

# chemistry utilities
from rdkit.Chem.rdchem import (
    Atom,
    Bond,
    Mol,
)
from rdkit.Chem.rdmolfiles import MolFragmentToSmarts

## Custom
from .linkers import anchor_and_linker_idxs
from .selection import (
    AtomCondition,
    BondCondition,
    logical_or,
    all_atoms,
    atom_neighbors_by_condition,
    bonds_by_condition,
    bond_condition_by_atom_condition_factory
)
from ...geometry.arraytypes import Shape, N
from ...mupr.connection import Connector


# Representation component initializers
def chemical_graph_from_rdkit(
    rdmol : Mol,
    atom_condition : Optional[AtomCondition]=None,
    label_method : Callable[[Atom], Hashable]=lambda atom : atom.GetIdx(),
    binary_operator : Callable[[bool, bool], bool]=logical_or,
    graph_type : Type[GraphLike]=nx.Graph,
) -> GraphLike:
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

    return graph_type(
        (label_method(atom_begin), label_method(atom_end))
            for (atom_begin, atom_end) in bonds_by_condition(
                rdmol,
                condition=bond_condition,
                as_pairs=True,    # return bond as pair of atoms,
                as_indices=False, # ...each as Atom objects
                negate=False,
            )
    )

def atom_positions_from_rdkit(
    rdmol : Mol, 
    conformer_idx : Optional[int]=None, 
    atom_idxs : Optional[Iterable[int]]=None,
) -> Optional[np.ndarray[Shape[N, 3], float]]:
    '''Boilerplate for fetching a subset of atom positions (if conformer it set) from an RDKit Mol'''
    if conformer_idx is None:
        return None
    
    if atom_idxs is None:
        atom_idxs = (atom.GetIdx() for atom in rdmol.GetAtoms())

    conformer = rdmol.GetConformer(conformer_idx) # DEVNOTE: will raise Exception if bad ID is provided; no need to check locally
    # return conformer.GetPositions()[[idx for idx in atom_idxs], :] 

    atom_positions = tuple(
        np.array(conformer.GetAtomPosition(atom_idx), dtype=float)
            for atom_idx in atom_idxs
    )
    if atom_positions:
        return np.vstack(atom_positions)
    return None # making explicit just to clarify that None return can still happen at this stage

def connector_between_rdatoms(
    parent_mol : Mol,
    from_atom_idx : int,
    to_atom_idx : int,
    conformer_idx : Optional[int]=None,
    anchor_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    linker_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    connector_labeller : Optional[Callable[[Connector], Hashable]]=None,
) -> Connector:
    '''
    Create a Connector object representing a (one-way) connection between two RDKit atoms
    
    Parameters
    ----------
    parent_mol : Mol
        The RDKit Mol containing the atoms of interest
    from_atom_idx : int
        The index of the "anchor" atom of the pair (on which the Connector will be centered)
    to_atom_idx : int
        The index of the "linker" atom of the pair (the neighbor atom of the anchor)
    anchor_labeller : Callable[[Atom], int], default: the atom index
        A function which takes an RDKit Atom object and returns a label for the anchor atom
    linker_labeller : Callable[[Atom], int], default: the atom index
        A function which takes an RDKit Atom object and returns a label for the linker atom
    conformer_idx : Optional[int], optional, default None
        The ID of the conformer from which to extract 3D positions
        If provided as None, will leave all spatial fields of the Connector unset
    connector_labeller : Callable[[Connector], Hashable], default: the Connector's DEFAULT_LABEL
        A function which takes a Connector object and returns a label for it
        This is called after all other fields of the Connector have been set
        (i.e. can make use of those fields in determination of the label)
        
    Returns
    -------
    connector : Connector
        The initialized Connector object
    '''
    if connector_labeller is None:
        connector_labeller = lambda conn : Connector.DEFAULT_LABEL

    # extract RDKit components - NOTE: will raise Exception immediately if pair of atoms are not, in fact, bonded
    bond : Bond = parent_mol.GetBondBetweenAtoms(from_atom_idx, to_atom_idx)
    anchor_atom : Atom = parent_mol.GetAtomWithIdx(from_atom_idx)
    anchor_label = anchor_labeller(anchor_atom)
    linker_atom : Atom = parent_mol.GetAtomWithIdx(to_atom_idx)
    linker_label = linker_labeller(linker_atom)

    # initialize Connector object
    connector = Connector(
        anchor=anchor_label,
        linker=linker_label,
        linkables={linker_label}, # register linker as bondable by default
        bondtype=bond.GetBondType(),
        query_smarts=MolFragmentToSmarts(
            parent_mol,
            atomsToUse=[from_atom_idx, to_atom_idx],
            bondsToUse=[bond.GetIdx()],
        ),
        metadata={
            'bond_stereo' : bond.GetStereo(),
            'bond_stereo_atoms' : tuple(bond.GetStereoAtoms()),
            **bond.GetPropsAsDict(includePrivate=True, includeComputed=False), # NOTE: computed props suppressed to avoid "unpicklable RDKit vector" errors
        }
    )
    connector.label = connector_labeller(connector)

    ## inject spatial info, if present
    connector_positions = atom_positions_from_rdkit(
        parent_mol,
        conformer_idx=conformer_idx,
        atom_idxs=[from_atom_idx, to_atom_idx]
    )
    if connector_positions is not None:
        connector.anchor_position = connector_positions[0, :]
        connector.linker_position = connector_positions[1, :]

        # define dihedral plane by neighbor atom, if a suitable one is present
        non_linker_nb_atom_positions = atom_positions_from_rdkit(
            parent_mol,
            conformer_idx=conformer_idx,
            atom_idxs=atom_neighbors_by_condition(
                anchor_atom,
                condition=lambda neighbor : (neighbor.GetIdx() == to_atom_idx),
                negate=True, # ensure the tangent point is not the linker itself
                as_indices=True,
            )
        )
        if non_linker_nb_atom_positions is not None:
            connector.set_dihedral_from_coplanar_point(non_linker_nb_atom_positions[0, :])

    return connector

def connectors_from_rdkit(
    rdmol : Mol,
    conformer_idx : Optional[int]=None,
    **kwargs,
) -> Generator['Connector', None, None]:
    '''Determine all Connectors contained in an RDKit Mol, as specified by wild-type linker atoms'''
    rdmol.UpdatePropertyCache() # avoids implicitValence errors on substructure match
    for (anchor_idx, linker_idx) in anchor_and_linker_idxs(rdmol):
        yield connector_between_rdatoms(
            rdmol,
            from_atom_idx=anchor_idx,
            to_atom_idx=linker_idx,
            conformer_idx=conformer_idx,
            **kwargs,
        )
