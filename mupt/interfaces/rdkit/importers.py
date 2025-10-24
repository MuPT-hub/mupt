'''Readers which convert RDKit Atoms and Mols into the MuPT molecular representation'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

from typing import (
    Hashable,
    Optional,
)

from rdkit.Chem.rdchem import (
    Atom,
    Mol,
    Conformer,
    StereoInfo,
)
from rdkit.Chem.rdmolops import FindPotentialStereo, GetMolFrags
from rdkit.Chem.rdDistGeom import EmbedMolecule

from ...chemistry.linkers import is_linker
from .components import atom_positions_from_rdkit, connector_between_rdatoms

from .labelling import name_for_rdkit_mol
from ...geometry.shapes import PointCloud
from ...chemistry.smiles import DEFAULT_SMILES_WRITE_PARAMS, SmilesWriteParams
from ...chemistry.conversion import rdkit_atom_to_element

from ...mupr.primitives import Primitive, PrimitiveHandle
from ...mupr.connection import TraversalDirection


def primitive_from_rdkit_atom(
    parent_mol : Mol,
    atom_idx : int,
    conformer_idx : Optional[int]=None,
    attach_connectors : bool=False,
    **kwargs
) -> Primitive:
    '''Initialize an atomic Primitive from an RDKit Atom'''
    atom : Atom = parent_mol.GetAtomWithIdx(atom_idx)
    atom_primitive = Primitive(
        element=rdkit_atom_to_element(atom),
        label=atom_idx,
        metadata=atom.GetPropsAsDict(
            includePrivate=True,
            includeComputed=False, # NOTE: computed props suppressed to avoid "unpicklable RDKit vector" errors 
        ), 
    )
    if (map_num := atom.GetAtomMapNum()) != 0:
        atom_primitive.metadata['molAtomMapNumber'] = map_num
    
    atom_pos = atom_positions_from_rdkit(parent_mol, conformer_idx=conformer_idx, atom_idxs=[atom_idx])
    if atom_pos is not None:
        atom_primitive.shape = PointCloud(positions=atom_pos[0, :]) # extract as vector from 2D array
    
    if attach_connectors:
        for nb_atom in atom.GetNeighbors(): # TODO: decide how bond Props should be split among metadata of the two bonded atoms
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
    **kwargs,
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
        label = name_for_rdkit_mol(rdmol_chain, smiles_writer_params=smiles_writer_params)
    rdmol_primitive = Primitive(
        label=label,
        metadata=rdmol_chain.GetPropsAsDict(includePrivate=True, includeComputed=False)
    )
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
            **kwargs,
        )
        begin_conn_handle = begin_prim.register_connector(begin_conn)
        rdmol_primitive.bind_external_connector(begin_prim_handle, begin_conn_handle, label=external_linker_label)
        
        ### Primitive 2 + associated Connector
        end_prim_handle = atom_idx_to_handle_map[end_idx]
        end_prim = rdmol_primitive.fetch_child(end_prim_handle)
        end_conn = connector_between_rdatoms(
            rdmol_chain,
            from_atom_idx=end_idx,
            to_atom_idx=begin_idx,
            conformer_idx=conformer_idx,
            **kwargs,
        )
        end_conn_handle = end_prim.register_connector(end_conn)
        rdmol_primitive.bind_external_connector(end_prim_handle, end_conn_handle, label=external_linker_label)
        
        ### joining of the pair of Connectors
        rdmol_primitive.connect_children(
            begin_prim_handle,
            begin_conn_handle,
            end_prim_handle,
            end_conn_handle,
        )

    # 3) excise temporary linker Primitives no longer needed as doorstops
    for linker_idx in linker_idxs:
        rdmol_primitive.detach_child(atom_idx_to_handle_map[linker_idx])

    ## 3a) insert traversal direction info based on 1-2 map number convention
    for ext_conn_handle, conn_ref in rdmol_primitive.external_connectors.items():
        atom_primitive = rdmol_primitive.fetch_child(conn_ref.primitive_handle)
        ext_conn = rdmol_primitive.fetch_connector(ext_conn_handle)
        
        if (mapnum := atom_primitive.metadata.get('molAtomMapNumber')) in {1,2}:
            chain_direction = TraversalDirection(mapnum)
            ext_conn.anchor.attachables.add(chain_direction)
            ext_conn.linker.attachables.add(TraversalDirection.complement(chain_direction))

    # 4) Inject conformer info - DEV: there are many avenues to do this (e.g. collate shape from children, if not None on all), but opted for the simplest for now
    non_linker_conformer = atom_positions_from_rdkit(
        rdmol_chain,
        conformer_idx=conformer_idx,
        atom_idxs=sorted(atom_idx_to_handle_map.keys() - linker_idxs), # preserve atom order
    )
    if non_linker_conformer is not None: # can't just check if Falsy in case this is an array (would need all() then)
        rdmol_primitive.shape = PointCloud(positions=non_linker_conformer) # exploit default NoneType value
    rdmol_primitive.check_self_consistent()
        
    return rdmol_primitive
    
def primitive_from_rdkit(
    rdmol : Mol,
    conformer_idx : Optional[int]=None,
    label : Optional[Hashable]=None,
    smiles_writer_params : SmilesWriteParams=DEFAULT_SMILES_WRITE_PARAMS,
    sanitize_frags : bool=True,
    denest : bool=True,
    **kwargs,
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
            **kwargs,
        )
    # otherwise, bind Primitives for each chain to "universal" root Primitive
    else:
        universe_primitive = Primitive(
            label=label
            # DEV: deliberately excluding metadata here to avoid squashing that of individual chains
        ) 
        for chain in chains:
            universe_primitive.attach_child(
                primitive_from_rdkit_chain(
                    chain,
                    conformer_idx=conformer_idx,
                    label=None, # impose default label for each individual chain
                    smiles_writer_params=smiles_writer_params,
                    **kwargs,
                )
            )
        return universe_primitive