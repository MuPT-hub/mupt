'''Interfaces between MuPT and RDKit Mols'''

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
    RWMol,
    Conformer,
    StereoInfo,
)
from rdkit.Chem.rdmolops import (
    FindPotentialStereo,
    GetMolFrags,
)
from rdkit.Chem.rdmolfiles import (
    MolToSmiles,
    MolFragmentToSmarts,
    SmilesWriteParams,
)
from rdkit.Chem.rdDistGeom import EmbedMolecule

## Custom
from ..chemistry.linkers import (
    is_linker,
    anchor_and_linker_idxs,
)
from ..chemistry.selection import (
    AtomCondition,
    BondCondition,
    logical_or,
    all_atoms,
    atom_neighbors_by_condition,
    bonds_by_condition,
    bond_condition_by_atom_condition_factory
)

# Representation components
from . import DEFAULT_SMILES_WRITE_PARAMS
from ..chemistry.core import ELEMENTS, ElementLike

from ..geometry.shapes import PointCloud
from ..geometry.arraytypes import Shape, N

from ..mupr.connection import Connector
from ..mupr.primitives import Primitive, PrimitiveHandle


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
    # return conformer.GetPositions()[[idx for idx in atom_idxs], :] # DEV: commenting out to avoid need list unpack + potential pivot to mapping as output
    return np.vstack([
        np.array(conformer.GetAtomPosition(atom_idx), dtype=float)
            for atom_idx in atom_idxs
    ])

def connector_between_rdatoms(
    parent_mol : Mol,
    from_atom_idx : int,
    to_atom_idx : int,
    conformer_idx : Optional[int]=None,
    from_atom_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    to_atom_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
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
    from_atom_labeller : Callable[[Atom], int], default: the atom index
        A function which takes an RDKit Atom object and returns a label for the anchor atom
    to_atom_labeller : Callable[[Atom], int], default: the atom index
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
    anchor_label = from_atom_labeller(anchor_atom)
    linker_atom : Atom = parent_mol.GetAtomWithIdx(to_atom_idx)
    linker_label = to_atom_labeller(linker_atom)

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
            **bond.GetPropsAsDict(),
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

            
# RDKit parsers
# Import from RDKit
def primitive_from_rdkit_atom(
    parent_mol : Mol,
    atom_idx : int,
    conformer_idx : Optional[int]=None,
    attach_connectors : bool=False,
    **kwargs
) -> Primitive:
    '''Initialize an atomic Primitive from an RDKit Atom'''
    atom : Atom = parent_mol.GetAtomWithIdx(atom_idx)
    
    elem : ElementLike = ELEMENTS[atom.GetAtomicNum()]
    if (mass_number := atom.GetIsotope()) != 0:
        elem = elem[mass_number] # fetch Isotope instance
    if (charge := atom.GetFormalCharge()) != 0:
        elem = elem.ion[charge] # fetch Ion instance - NOTE: order here is deliberate; can't fetch Isotope of Ion, but CAN fetch Ion of Isotope
    
    atom_primitive = Primitive(
        element=elem,
        label=atom_idx,
        metadata=atom.GetPropsAsDict(includePrivate=True),
    )
    atom_pos = atom_positions_from_rdkit(parent_mol, conformer_idx=conformer_idx, atom_idxs=[atom_idx])
    if atom_pos is not None:
        atom_pos = atom_pos[0, :] # extract single position from 2D array
        atom_primitive.shape = PointCloud(positions=atom_pos)
    
    if attach_connectors:
        for nb_atom in atom.GetNeighbors():
            conn_handle = atom_primitive.register_connector(
                connector_between_rdatoms(
                    parent_mol=parent_mol,
                    from_atom_idx=atom_idx,
                    to_atom_idx=nb_atom.GetIdx(),
                    conformer_idx=conformer_idx,
                    **kwargs,
                )
            )
    return atom_primitive
    
def primitive_from_rdkit_chain(
    rdmol_chain : Mol,
    conformer_idx : Optional[int]=None,
    label : Optional[Hashable]=None,
    atom_label : str='ATOM',
    external_linker_label : str='*',
    smiles_writer_params : SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
) -> Primitive:
    ''' 
    Initialize a Primitive hierarchy from an RDKit Mol representing a single molecule

    Parameters
    ----------
    rdmol : Chem.Mol
        The RDKit Mol object to convert
    conformer_idx : int, optional
        The ID of the conformer to use, by default None (uses no conformer)
    label : Hashable, optional
        A distinguishing label for the Primitive
        If none is provided, the canonical SMILES of the RDKit Mol will be used
    
    Returns
    -------
    Primitive
        The created Primitive object
    '''
    if label is None:
        label = MolToSmiles(rdmol_chain, params=smiles_writer_params)
    rdmol_primitive = Primitive(label=label)
    ## DEV: opting to not inject stereochemical metadata for now, since that may change as Primitive repr is transformed geometrically
    # stereo_info_map : dict[int, StereoInfo] = {
    #     stereo_info.centeredOn : stereo_info # TODO: determine most appropriate choice of flags to use in FindPotentialStereo
    #         for stereo_info in FindPotentialStereo(rdmol_chain, cleanIt=True, flagPossible=True) 
    # } 

    # 1) Insert child Primitives for each atom (EVEN linkers - this keeps indices in sync for final handle assignment)
    linker_idxs : set[int] = set()
    atom_idx_to_handle_map : dict[int, PrimitiveHandle] = dict() # DEV: as-implemented, handle idx **SHOULD** match atom idx, but it never hurts to be explicit :P
    for atom in rdmol_chain.GetAtoms(): # DEV: opting not to get atoms implicitly from bonds to handle single, unbonded atom (e.g. noble gas) uniformly
        atom_idx = atom.GetIdx()
        if is_linker(atom):
            linker_idxs.add(atom_idx)
        
        atom_prim = primitive_from_rdkit_atom(
            rdmol_chain,
            atom_idx,
            conformer_idx=conformer_idx,
            attach_connectors=False, # will attach per-bond to avoid needing to match connector handles to bond idxs
        )
        atom_idx_to_handle_map[atom_idx] = rdmol_primitive.attach_child(atom_prim, label=atom_label)
    
    # 2) forge connections between Primitives corresponding to bonded atoms (propagating external Connectors up to mol primitive)
    for bond in rdmol_chain.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        ## Primitive 1 + associated Connector
        begin_prim_handle = atom_idx_to_handle_map[begin_idx]
        begin_prim = rdmol_primitive.fetch_child(begin_prim_handle)
        begin_conn = connector_between_rdatoms(
            rdmol_chain,
            from_atom_idx=begin_idx,
            to_atom_idx=end_idx,
            conformer_idx=conformer_idx,
            # TODO: customize labelling?
        )
        begin_conn.linkables.add(external_linker_label)
        begin_conn_handle = begin_prim.register_connector(begin_conn)
        rdmol_primitive.bind_external_connector(begin_prim_handle, begin_conn_handle)
        
        ### Primitive 2 + associated Connector
        end_prim_handle = atom_idx_to_handle_map[end_idx]
        end_prim = rdmol_primitive.fetch_child(end_prim_handle)
        end_conn = connector_between_rdatoms(
            rdmol_chain,
            from_atom_idx=end_idx,
            to_atom_idx=begin_idx,
            conformer_idx=conformer_idx,
            # TODO: customize labelling?
        )
        end_conn.linkables.add(external_linker_label)
        end_conn_handle = end_prim.register_connector(end_conn)
        rdmol_primitive.bind_external_connector(end_prim_handle, end_conn_handle)
        
        ### joining of the pair of Connectors
        rdmol_primitive.connect_children(
            begin_prim_handle,
            end_prim_handle,
            begin_conn_handle,
            end_conn_handle,
        )

    # 3) excise temporary linker Primitives no longer needed as doorstops
    for linker_idx in linker_idxs:
        rdmol_primitive.detach_child(atom_idx_to_handle_map[linker_idx])

    # 4) Inject conformer info - DEV: there are many avenues to do this (e.g. collate shape from children, if not None on all), but opted for the simplest for now
    non_linker_conformer = atom_positions_from_rdkit(
        rdmol_chain,
        conformer_idx=conformer_idx,
        atom_idxs=sorted(atom_idx_to_handle_map.keys() - linker_idxs), # preserve atom order
    )
    rdmol_primitive.shape = non_linker_conformer or PointCloud(positions=non_linker_conformer) # exploit default NoneType value
    rdmol_primitive.check_self_consistent()
        
    return rdmol_primitive
    
def primitive_from_rdkit(
    rdmol : Mol,
    conformer_idx : Optional[int]=None,
    label : Optional[Hashable]=None,
    smiles_writer_params : SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
    sanitize_frags : bool=True,
    denest : bool=True,
) -> Primitive:
    '''
    Initialize a Primitive hierarchy from an RDKit Mol representing one or more molecules
    '''
    chains = GetMolFrags(
        rdmol,
        asMols=True, 
        sanitizeFrags=sanitize_frags,
        # DEV: leaving these None for now, but highlighting that we can spigot more info out of this eventually
        frags=None,
        fragsMolAtomMapping=None,
    )
    
    # if only 1 chain is present, fall back to single-chain importer
    if (len(chains) == 1) and denest:
        return primitive_from_rdkit_chain(
            chains[0],
            conformer_idx=conformer_idx,
            label=label,
            smiles_writer_params=smiles_writer_params,
        )
    # otherwise, bind Primitives for each chain to "universal" root Primitive
    else:
        universe_primitive = Primitive(label=label)
        for chain in chains:
            universe_primitive.attach_child(
                primitive_from_rdkit_chain(
                    chain,
                    conformer_idx=conformer_idx,
                    label=None, # impose default SMILES-based label for each individual chain
                    smiles_writer_params=smiles_writer_params,
                )
            )
        return universe_primitive
    
## Export to RDKit
def primitive_to_rdkit(primitive : Primitive) -> Mol:
    '''
    Convert a StructuralPrimitive to an RDKit Mol
    Will return as single Mol instance, even is underlying Primitive represents a collection of multiple disconnected molecules
    '''
    if not primitive.is_atomizable:
        raise ValueError('Cannot export Primitive with non-atomic parts to RDKit Mol')
    
    rwmol = RWMol()
    primitive.flatten()
    for atom_primitive in primitive.children:
        rdatom = Atom(atom_primitive.element.symbol)
        ...

    # DEV: model assembly off of OpenFF RDKit TK wrapper
    # https://github.com/openforcefield/openff-toolkit/blob/5b4941c791cd49afbbdce040cefeb23da298ada2/openff/toolkit/utils/rdkit_wrapper.py#L2330

    # handle identification of shape (if EVERY atom has a postiions and if those are consistent with the Primitive's shape, if PointCloud)

    # case 1 : export single atom to RDKit Atom
    
    # case 2 : link up atoms within strucutral primitive recursively
    ## match connectors along bonds, identify external connectors
