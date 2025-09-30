'''Interfaces between MuPT and RDKit Mols'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'


from typing import Callable, Generator, Hashable, Optional, Type, TypeVar, Union

import numpy as np
import networkx as nx
GraphLike = TypeVar('GraphLike', bound=nx.Graph)

# chemistry utilities
from periodictable import elements as ELEMENTS

from rdkit.Chem.rdchem import (
    Atom,
    Bond,
    Mol,
    RWMol,
    Conformer,
    StereoInfo,
)
from rdkit.Chem.rdmolops import (
    AddHs,
    FindPotentialStereo,
    GetMolFrags,
    SanitizeMol,
    SanitizeFlags,
    SANITIZE_ALL,
)
from rdkit.Chem.rdmolfiles import (
    MolFromSmiles,
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
from ..geometry.shapes import PointCloud
from ..mupr.connection import Connector
from ..mupr.primitives import Primitive, PrimitiveHandle

# module-scoped constants
DEFAULT_SMILES_WRITE_PARAMS = SmilesWriteParams()
DEFAULT_SMILES_WRITE_PARAMS.doIsomericSmiles = True
DEFAULT_SMILES_WRITE_PARAMS.doKekule         = False
DEFAULT_SMILES_WRITE_PARAMS.canonical        = True
DEFAULT_SMILES_WRITE_PARAMS.allHsExplicit    = False
DEFAULT_SMILES_WRITE_PARAMS.doRandom         = False


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

def connector_between_rdatoms(
    parent_mol : Mol,
    from_atom_idx : int,
    to_atom_idx : int,
    from_atom_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    to_atom_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    conformer_id : Optional[int]=None,
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
    conformer_id : Optional[int], optional, default None
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
    
    # check for conformer
    conformer : Optional[Conformer] = None
    if conformer_id is not None:
        conformer = parent_mol.GetConformer(conformer_id)

    # extract RDKit components - NOTE: will raise Exception immediately if pair of atoms are not, in fact, bonded
    bond : Bond = parent_mol.GetBondBetweenAtoms(from_atom_idx, to_atom_idx)
    bondtype = bond.GetBondType() 
    
    anchor_atom : Atom = parent_mol.GetAtomWithIdx(from_atom_idx)
    anchor_label = from_atom_labeller(anchor_atom)
    
    linker_atom : Atom = parent_mol.GetAtomWithIdx(to_atom_idx)
    linker_label = to_atom_labeller(linker_atom)

    # initialize Connector object
    connector = Connector(
        anchor=anchor_label,
        linker=linker_label,
        linkables={linker_label}, # register linker as bondable by default
        bondtype=bondtype,
        query_smarts=MolFragmentToSmarts(
            parent_mol,
            atomsToUse=[from_atom_idx, to_atom_idx],
            bondsToUse=[bond.GetIdx()],
        ),
    )
    connector.label = connector_labeller(connector)

    ## inject spatial info, if present
    if conformer:
        connector.anchor_position = np.array(conformer.GetAtomPosition(from_atom_idx), dtype=float)
        connector.linker_position = np.array(conformer.GetAtomPosition(to_atom_idx), dtype=float)

        # define dihedral plane by neighbor atom, if a suitable one is present
        real_neighbor_atom_idxs : Generator[int, None, None] = atom_neighbors_by_condition(
            anchor_atom,
            condition=lambda neighbor : (neighbor.GetIdx() == to_atom_idx),
            negate=True, # ensure the tangent point is not the linker itself
            as_indices=True,
        )
        try:
            ## TODO: offer option to make this more selective (i.e. choose which neighbor atom lies in the dihedral plane)
            connector.set_dihedral_from_coplanar_point(
                np.array(conformer.GetAtomPosition(next(real_neighbor_atom_idxs)), dtype=float),
            )
        except StopIteration:
            pass

    return connector

def connectors_from_rdkit(
    rdmol : Mol,
    conformer_id : Optional[int]=None,
    linker_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    anchor_labeller : Callable[[Atom], int]=lambda atom : atom.GetIdx(),
    connector_labeller : Optional[Callable[[Connector], Hashable]]=None,
) -> Generator['Connector', None, None]:
    '''Determine all Connectors contained in an RDKit Mol, as specified by wild-type linker atoms'''
    rdmol.UpdatePropertyCache() # avoids implicitValence errors on substructure match
    for (anchor_idx, linker_idx) in anchor_and_linker_idxs(rdmol):
        yield connector_between_rdatoms(
            rdmol,
            from_atom_idx=anchor_idx,
            to_atom_idx=linker_idx,
            from_atom_labeller=anchor_labeller,
            to_atom_labeller=linker_labeller,
            conformer_id=conformer_id,
            connector_labeller=connector_labeller,
        )

def shape_from_rdkit(
    rdmol : Mol, 
    conformer_id : Optional[int]=None, 
    atom_idxs: Optional[list[int]]=None,
) -> Optional[PointCloud]:
    '''Extract a PointCloud shape from an RDKit Mol, if possible'''
    if (conformer_id is None) or (rdmol.GetNumConformers() == 0):
        return None
    
    if atom_idxs is None:
        atom_idxs = [atom.GetIdx() for atom in rdmol.GetAtoms()]
    
    conformer = rdmol.GetConformer(conformer_id) # DEVNOTE: will raise Exception if bad ID is provided; no need to check ourselves
    positions : np.ndarray = conformer.GetPositions() 
    
    return PointCloud(positions[atom_idxs, :])
            
# Imports and Exporters
def primitive_from_rdkit(
    rdmol : Mol,
    conformer_id : Optional[int]=None,
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
        # DEV: leaving these None for now, but higlighting that we can spigot more info out of this eventually
        frags=None,
        fragsMolAtomMapping=None,
    )
    
    # if only 1 chain is present, fall back to single-chain importer
    if (len(chains) == 1) and denest:
        return primitive_from_rdkit_chain(
            chains[0],
            conformer_id=conformer_id,
            label=label, # impose default SMILES-based label at chain level at all times
        )
    # otherwise, bind Primitives for each chain to "universal" root Primitive
    else:
        universe_primitive = Primitive(label=label)
        for chain in chains:
            universe_primitive.attach_child(
                primitive_from_rdkit_chain(
                    chain,
                    conformer_id=conformer_id,
                    label=None,
                    smiles_writer_params=smiles_writer_params,
                )
            )
        return universe_primitive

def primitive_from_rdkit_chain(
    rdmol_chain : Mol,
    conformer_id : Optional[int]=None,
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
    conformer_id : int, optional
        The ID of the conformer to use, by default None (uses no conformer)
    label : Hashable, optional
        A distinguishing label for the Primitive
        If none is provided, the canonical SMILES of the RDKit Mol will be used
    
    Returns
    -------
    Primitive
        The created Primitive object
    '''
    # Extract information from the RDKit Mol
    ## TODO: check that rdmol_chain really has exactly 1 connected component?
    conformer : Optional[Conformer] = None
    if conformer_id is not None:
        conformer = rdmol_chain.GetConformer(conformer_id)

    stereo_info_map : dict[int, StereoInfo] = {
        stereo_info.centeredOn : stereo_info # TODO: determine most appropriate choice of flags to use in FindPotentialStereo
            for stereo_info in FindPotentialStereo(rdmol_chain, cleanIt=True, flagPossible=True) 
    }
    if label is None:
        label = MolToSmiles(rdmol_chain, params=smiles_writer_params)
    
    # Initialize molecule-level resolution "parent" Primitive
    rdmol_primitive = Primitive(label=label)
    
    ## insert subprimitives for all atoms (real or otherwise)
    linker_idxs : set[int] = set()
    atom_map : dict[int, PrimitiveHandle] = dict()
    for atom in rdmol_chain.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_primitive = Primitive(
            element=ELEMENTS[atom.GetAtomicNum()], # NOTE: this is part of what necessitates excluding atomic number 0 linkers 
            label=atom_idx,
            metadata={
                **atom.GetPropsAsDict(includePrivate=True),
                'stereo_info' : stereo_info_map.get(atom_idx, None) # TODO: see if there's a more local way of getting at stereo_info
            }, 
        )
        if conformer is not None:
            atom_primitive.shape = PointCloud(np.array(conformer.GetAtomPosition(atom_idx)))
            
        atom_map[atom_idx] = rdmol_primitive.attach_child(atom_primitive, label=atom_label)
        if is_linker(atom): # NOTE: attaching linkers as if they were Primitives first to ensure handle index matches atom index
            linker_idxs.add(atom_idx)
    
    ## forge connections between atoms, and with external connections
    for bond in rdmol_chain.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx   = bond.GetEndAtomIdx()
        bonded_idxs : set[int] = {begin_idx, end_idx}
        
        linker_atom_idxs : set[int] = linker_idxs.intersection(bonded_idxs)
        if len(linker_atom_idxs) > 1:
            raise ValueError(f'Illegal multi-linker bond between atoms {bonded_idxs}')
        
        ### external (i.e. off-molecule) bond
        elif len(linker_atom_idxs) == 1: 
            anchor_atom_idxs = bonded_idxs - linker_atom_idxs
            anchor_atom_idx = anchor_atom_idxs.pop()
            linker_atom_idx = linker_atom_idxs.pop()

            connector_external = connector_between_rdatoms(
                rdmol_chain,
                from_atom_idx=anchor_atom_idx,
                to_atom_idx=linker_atom_idx,
                conformer_id=conformer_id,
            )
            connector_external_toplevel = connector_external.copy() # the version of the connector propagated up to the molecular level
            connector_external_toplevel.linkables.add(external_linker_label) # include distinguishing label for off-molecule version of this Connector

            anchor_prim_handle = atom_map[anchor_atom_idx]
            anchor_atom_prim = rdmol_primitive.fetch_child(anchor_prim_handle)
            rdmol_primitive.pair_external_connectors_vertically(
                rdmol_primitive.register_connector(connector_external),
                anchor_prim_handle,
                anchor_atom_prim.register_connector(connector_external_toplevel),
            )
            
        ### internal bond between pair of "real" atoms
        elif len(linker_atom_idxs) == 0: 
            atom_prim_handle_1 = atom_map[begin_idx]
            atom_prim_1 = rdmol_primitive.fetch_child(atom_prim_handle_1)
            connector_internal_1_handle = atom_prim_1.register_connector(
                connector_between_rdatoms(
                    rdmol_chain,
                    from_atom_idx=begin_idx,
                    to_atom_idx=end_idx,
                    conformer_id=conformer_id,
                )
            )
                
            atom_prim_handle_2 = atom_map[end_idx]
            atom_prim_2 = rdmol_primitive.fetch_child(atom_prim_handle_2)
            connector_internal_2_handle = atom_prim_2.register_connector(
                connector_between_rdatoms(
                    rdmol_chain,
                    from_atom_idx=end_idx,
                    to_atom_idx=begin_idx,
                    conformer_id=conformer_id,
                )
            )
            
            rdmol_primitive.link_children(
                atom_prim_handle_1,
                atom_prim_handle_2,
                connector_internal_1_handle,
                connector_internal_2_handle,
            )

    ## excise temporary linker Primitives no longer needed for bookkeeping
    for linker_idx in linker_idxs:
        rdmol_primitive.detach_child(atom_map.pop(linker_idx)) # pops both from atom_map AND molecule Primitive's children
        
    # Inject information into molecule-level Primitive now that atoms have been sorted out
    if conformer is not None:
        rdmol_primitive.shape = PointCloud(
            positions=np.vstack(
                [atom_prim.shape.positions for atom_prim in rdmol_primitive.children]
            )
        )
        
    return rdmol_primitive
    
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

# SMILES/SMARTS readers and writers
def primitive_from_smiles(
        smiles : str, 
        ensure_explicit_Hs : bool=True,
        embed_positions : bool=False,
        sanitize_ops : SanitizeFlags=SANITIZE_ALL,
        label : Optional[Hashable]=None,
    ) -> Primitive:
    '''Create a Primitive from a SMILES string, optionally embedding positions if selected'''
    rdmol = MolFromSmiles(smiles, sanitize=False)
    if ensure_explicit_Hs:
        rdmol.UpdatePropertyCache() # allow Hs to be added without sanitizating twice
        rdmol = AddHs(rdmol)
    SanitizeMol(rdmol, sanitizeOps=sanitize_ops)
    
    conformer_id : Optional[int] = None
    if embed_positions:
        conformer_id = EmbedMolecule(rdmol, clearConfs=False) # NOTE: don't clobber existing conformers for safety (though new Mol shouldn't have any anyway)
    
    return primitive_from_rdkit(rdmol, conformer_id=conformer_id, label=label)